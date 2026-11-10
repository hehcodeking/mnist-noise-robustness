"""Evaluate saved checkpoints once under the fixed test protocol."""
import argparse
import csv
import hashlib
import json
import numpy as np
import torch
from data import ROOT, load_data, normalized, gaussian
from models import make_model
from metrics import classification_metrics, paired_bootstrap_delta


@torch.inference_mode()
def predict(model, images, device):
    parts = []
    for start in range(0, len(images), 512):
        logits = model(torch.from_numpy(images[start:start + 512]).to(device))
        parts.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((ROOT / "config.json").read_text())
    _, _, x_test, y = load_data()
    x = normalized(x_test)
    noise = np.random.default_rng(config["test_noise_seed"]).standard_normal(x.shape, dtype=np.float32)
    uniform = np.random.default_rng(config["test_noise_seed"] + 1).random(x.shape, dtype=np.float32)
    p = config["salt_pepper_probability"]
    sp = np.where(uniform < p / 2, 0, np.where(uniform > 1 - p / 2, 1, x)).astype(np.float32)
    results = ROOT / "results"
    predictions = {"labels": y}
    rows = []
    for name in config["conditions"]:
        for seed in config["seeds"]:
            path = ROOT / "checkpoints" / f"{name}_{seed}.pt"
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            model = make_model(ckpt["architecture"]).to(args.device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            conditions = [(f"gaussian_{s:.1f}", gaussian(x, s, noise)) for s in config["test_sigmas"]]
            conditions.append(("salt_pepper_0.1", sp))
            for corruption, images in conditions:
                pred = predict(model, images, args.device)
                predictions[f"{name}_{seed}_{corruption}"] = pred.astype(np.uint8)
                row = {"condition": name, "seed": seed, "corruption": corruption,
                       **classification_metrics(y, pred), "best_epoch": ckpt["best_epoch"],
                       "training_seconds": ckpt["training_seconds"], "parameters": ckpt["parameters"]}
                rows.append(row)
                print(json.dumps(row), flush=True)
    with (results / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(results / "predictions.npz", **predictions)
    contrasts = {}
    for seed in config["seeds"]:
        for corruption in ["gaussian_0.0", "gaussian_0.3", "gaussian_0.5", "salt_pepper_0.1"]:
            a = predictions[f"cnn_clean_{seed}_{corruption}"]
            b = predictions[f"cnn_noise030_{seed}_{corruption}"]
            contrasts[f"{seed}_{corruption}"] = paired_bootstrap_delta(y, a, b)
    (results / "paired_bootstrap.json").write_text(json.dumps(contrasts, indent=2))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (ROOT / "checkpoints").glob("*.pt")}
    (results / "checkpoint_sha256.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
