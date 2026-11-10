"""Train all predefined conditions. Test labels are not read by this script."""
from pathlib import Path
import argparse
import csv
import json
import os
import platform
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from data import ROOT, prepare_data, read_idx, split_indices, normalized, gaussian
from models import make_model


@torch.inference_mode()
def accuracy(model, images, labels, device):
    model.eval()
    correct = 0
    for start in range(0, len(labels), 512):
        logits = model(torch.from_numpy(images[start:start + 512]).to(device))
        correct += int((logits.argmax(1).cpu() == labels[start:start + 512]).sum())
    return correct / len(labels)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--conditions", nargs="*")
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    torch.set_num_threads(4)
    if args.device == "cpu":
        torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    prepare_data()
    x = read_idx("train-images-idx3-ubyte.gz")
    y = read_idx("train-labels-idx1-ubyte.gz").astype(np.int64)
    train_ids, val_ids = split_indices(y, config)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    np.savez(results / "split_indices.npz", train=train_ids, validation=val_ids)
    x_train = torch.from_numpy(normalized(x[train_ids])).to(device)
    y_train = torch.from_numpy(y[train_ids]).to(device)
    x_val = normalized(x[val_ids])
    y_val = torch.from_numpy(y[val_ids])
    val_noise = np.random.default_rng(config["validation_noise_seed"]).standard_normal(x_val.shape, dtype=np.float32)
    x_val_noise = gaussian(x_val, config["validation_sigma"], val_noise)
    environment = {"python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
                   "platform": platform.platform(), "architecture": platform.machine(),
                   "processor": platform.processor(), "device": str(device), "threads": torch.get_num_threads(),
                   "mps_available": torch.backends.mps.is_available(), "date": "2026-11-10",
                   "training_examples": len(train_ids), "validation_examples": len(val_ids)}
    (results / "environment.json").write_text(json.dumps(environment, indent=2))
    conditions = args.conditions or list(config["conditions"])
    for seed in args.seeds or config["seeds"]:
        for name in conditions:
            spec = config["conditions"][name]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = make_model(spec["architecture"]).to(device)
            # Separate RNGs preserve identical CNN initial weights and batch order
            # across augmentation conditions for a given seed.
            order_rng = torch.Generator().manual_seed(seed + 10000)
            aug_rng = torch.Generator().manual_seed(seed + 20000)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            loss_fn = nn.CrossEntropyLoss()
            rows, best_score, best_epoch, best_state = [], -1.0, 0, None
            synchronize(device)
            start_time = time.perf_counter()
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                order = torch.randperm(len(train_ids), generator=order_rng)
                total_loss = 0.0
                for begin in range(0, len(order), config["batch_size"]):
                    ids = order[begin:begin + config["batch_size"]].to(device)
                    inputs, targets = x_train[ids], y_train[ids]
                    if spec["train_sigma"] > 0:
                        mask = (torch.rand((len(ids), 1, 1, 1), generator=aug_rng) < config["augmentation_probability"]).to(device)
                        noise = torch.randn(inputs.shape, generator=aug_rng).to(device)
                        inputs = torch.clamp(inputs + mask * noise * spec["train_sigma"], 0, 1)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(inputs), targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(ids)
                clean_acc = accuracy(model, x_val, y_val, device)
                noisy_acc = accuracy(model, x_val_noise, y_val, device)
                score = (clean_acc + noisy_acc) / 2
                if score > best_score:
                    best_score, best_epoch = score, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                row = {"seed": seed, "condition": name, "epoch": epoch,
                       "train_loss": total_loss / len(train_ids), "val_clean_accuracy": clean_acc,
                       "val_noise_accuracy": noisy_acc, "selection_score": score}
                rows.append(row)
                print(json.dumps(row), flush=True)
            synchronize(device)
            elapsed = time.perf_counter() - start_time
            checkpoint = {"state_dict": best_state, "architecture": spec["architecture"],
                          "seed": seed, "condition": name, "best_epoch": best_epoch,
                          "best_validation_score": best_score, "training_seconds": elapsed,
                          "parameters": sum(p.numel() for p in model.parameters()), "config": config}
            torch.save(checkpoint, ROOT / "checkpoints" / f"{name}_{seed}.pt")
            with (results / f"history_{name}_{seed}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"FINISHED {name} seed={seed} best_epoch={best_epoch} seconds={elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
