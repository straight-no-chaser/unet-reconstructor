#!/usr/bin/env python
"""Build or validate processed electrophysiology reconstruction arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import compute_apd_widths, extract_pairs_from_h5_directory


def describe_npy(path):
    arr = np.load(path, mmap_mode="r")
    return {"path": str(path), "shape": tuple(arr.shape), "dtype": str(arr.dtype)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/recording files", help="Directory containing original .h5 recordings")
    parser.add_argument("--output-dir", default="data/processed/inputs/training inputs", help="Directory for processed .npy arrays")
    parser.add_argument("--from-h5", action="store_true", help="Rebuild arrays from raw .h5 files using notebook logic")
    parser.add_argument("--compute-apd", action="store_true", help="Recompute APD widths from intras_new_normalization.npy")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_h5:
        extras, intras, apds = extract_pairs_from_h5_directory(args.raw_dir, output_dir)
        print(f"Wrote extras {extras.shape}, intras {intras.shape}, APDs {apds.shape} to {output_dir}")
        return

    if args.compute_apd:
        intra_path = output_dir / "intras_new_normalization.npy"
        if not intra_path.exists():
            raise FileNotFoundError(intra_path)
        intras = np.load(intra_path)
        apds = compute_apd_widths(intras)
        np.save(output_dir / "APDS_new_normalization.npy", apds)
        print(f"Wrote APDS_new_normalization.npy with shape {apds.shape}")

    candidates = sorted(output_dir.glob("*.npy"))
    if not candidates:
        print(f"No processed .npy files found in {output_dir}. Use --from-h5 when raw recordings are available.")
        return

    print("Processed arrays:")
    for info in [describe_npy(path) for path in candidates]:
        print(f"- {info['path']}: shape={info['shape']} dtype={info['dtype']}")


if __name__ == "__main__":
    main()
