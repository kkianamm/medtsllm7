# scripts/prepare_anomaly_data.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_numeric_csv(path: Path, drop_timestamp: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)

    if drop_timestamp:
        for col in list(df.columns):
            if "timestamp" in col.lower() or col.lower() in {"time", "date"}:
                df = df.drop(columns=[col])

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def prepare_psm(raw_dir: Path, out_dir: Path) -> None:
    """
    Expected raw files:
      raw_dir/train.csv
      raw_dir/test.csv
      raw_dir/test_label.csv

    Output files expected by datasets/psm.py:
      data/psm/train.csv
      data/psm/test.csv
      data/psm/test_label.csv
    """
    _ensure_dir(out_dir)

    train = _clean_numeric_csv(raw_dir / "train.csv", drop_timestamp=False)
    test = _clean_numeric_csv(raw_dir / "test.csv", drop_timestamp=False)
    label = pd.read_csv(raw_dir / "test_label.csv")

    # Keep timestamp column if repo's psm.py expects to drop it.
    # If missing, add dummy timestamp_(min).
    for df in (train, test, label):
        if "timestamp_(min)" not in df.columns:
            df.insert(0, "timestamp_(min)", np.arange(len(df)))

    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    # Label should have timestamp_(min) plus one anomaly column.
    if label.shape[1] > 2:
        timestamp = label["timestamp_(min)"] if "timestamp_(min)" in label.columns else np.arange(len(label))
        y = label.drop(columns=["timestamp_(min)"], errors="ignore").max(axis=1).astype(int)
        label = pd.DataFrame({"timestamp_(min)": timestamp, "label": y})

    label.to_csv(out_dir / "test_label.csv", index=False)

    print(f"Prepared PSM data in {out_dir}")


def prepare_msl(raw_dir: Path, out_dir: Path) -> None:
    """
    Expected raw files can be either:
      raw_dir/MSL_train.npy
      raw_dir/MSL_test.npy
      raw_dir/MSL_test_label.npy

    or:
      raw_dir/train.npy
      raw_dir/test.npy
      raw_dir/test_label.npy

    Output files expected by datasets/msl.py:
      data/msl/MSL_train.npy
      data/msl/MSL_test.npy
      data/msl/MSL_test_label.npy
    """
    _ensure_dir(out_dir)

    candidates = {
        "MSL_train.npy": ["MSL_train.npy", "train.npy"],
        "MSL_test.npy": ["MSL_test.npy", "test.npy"],
        "MSL_test_label.npy": ["MSL_test_label.npy", "test_label.npy", "labels.npy"],
    }

    for out_name, names in candidates.items():
        src = None
        for name in names:
            p = raw_dir / name
            if p.exists():
                src = p
                break

        if src is None:
            raise FileNotFoundError(
                f"Could not find any of {names} in {raw_dir}"
            )

        arr = np.load(src)
        arr = np.nan_to_num(arr)

        if out_name == "MSL_test_label.npy":
            arr = arr.astype(int)
            if arr.ndim > 1:
                arr = arr.max(axis=-1)

        np.save(out_dir / out_name, arr)

    print(f"Prepared MSL data in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["psm", "msl"])
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)

    args = parser.parse_args()

    if args.dataset == "psm":
        prepare_psm(args.raw_dir, args.out_dir)
    elif args.dataset == "msl":
        prepare_msl(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
