#!/usr/bin/env python3
"""Download MNIST and extract all samples as PNGs."""

import gzip
import struct
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm

TMP_DIR = Path("/tmp/mnist_raw")
OUT_DIR = Path(__file__).resolve().parent / "mnist"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
SPLITS = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def download(filename: str) -> Path:
    path = TMP_DIR / filename
    if path.exists():
        return path
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(BASE_URL + filename, path)
    return path


def read_labels(path: Path) -> list[int]:
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return list(f.read(n))


def read_images(path: Path) -> tuple[int, int, int, bytes]:
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return n, rows, cols, f.read()


def extract_split(split: str, img_file: str, lbl_file: str) -> None:
    labels = read_labels(download(lbl_file))
    n, rows, cols, data = read_images(download(img_file))
    img_size = rows * cols

    for i in tqdm(range(n), desc=split, unit="img"):
        out_dir = OUT_DIR / split / str(labels[i])
        out_dir.mkdir(parents=True, exist_ok=True)
        offset = i * img_size
        img = Image.frombytes("L", (cols, rows), data[offset : offset + img_size])
        img.save(out_dir / f"{i:05d}.png")


def main() -> None:
    for split, (img_file, lbl_file) in SPLITS.items():
        extract_split(split, img_file, lbl_file)
    print(f"\nDone. Images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
