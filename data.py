"""MNIST IDX reader, verified download, fixed stratified split and corruptions."""
from pathlib import Path
import gzip
import hashlib
import json
import struct
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def prepare_data():
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    manifest = {}
    for filename, md5 in FILES.items():
        path = folder / filename
        if not path.exists():
            print("Downloading", filename, flush=True)
            with urllib.request.urlopen(SOURCE + filename, timeout=120) as response:
                payload = response.read()
            if hashlib.md5(payload).hexdigest() != md5:
                raise ValueError("Download checksum mismatch: " + filename)
            path.write_bytes(payload)
        payload = path.read_bytes()
        if hashlib.md5(payload).hexdigest() != md5:
            raise ValueError("Checksum mismatch: " + filename)
        manifest[filename] = {"url": SOURCE + filename, "md5": md5,
                              "sha256": hashlib.sha256(payload).hexdigest(),
                              "bytes": len(payload)}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_idx(name):
    with gzip.open(ROOT / "data" / name, "rb") as stream:
        payload = stream.read()
    magic = struct.unpack(">I", payload[:4])[0]
    ndim = magic & 255
    assert (magic >> 8) == 8 and ndim in (1, 3)
    shape = struct.unpack(">" + "I" * ndim, payload[4:4 + 4 * ndim])
    array = np.frombuffer(payload, dtype=np.uint8, offset=4 + 4 * ndim).copy()
    return array.reshape(shape)


def load_data():
    prepare_data()
    return (read_idx("train-images-idx3-ubyte.gz"),
            read_idx("train-labels-idx1-ubyte.gz"),
            read_idx("t10k-images-idx3-ubyte.gz"),
            read_idx("t10k-labels-idx1-ubyte.gz"))


def split_indices(labels, config):
    """Never use the official test set to select indices or hyperparameters."""
    rng = np.random.default_rng(config["split_seed"])
    train, val = [], []
    n, v = config["train_per_class"], config["val_per_class"]
    for digit in range(10):
        ids = rng.permutation(np.flatnonzero(labels == digit))
        train.extend(ids[:n])
        val.extend(ids[n:n + v])
    return rng.permutation(train), rng.permutation(val)


def normalized(images):
    return images.astype(np.float32)[:, None] / 255.0


def gaussian(images, sigma, noise):
    return np.clip(images + np.float32(sigma) * noise, 0, 1)


if __name__ == "__main__":
    arrays = load_data()
    print([a.shape for a in arrays])
