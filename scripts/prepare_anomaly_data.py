# scripts/prepare_anomaly_data.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import wfdb


"""
Prepare MIT-BIH Arrhythmia Database for MedTsLLM anomaly_detection.

Input folder example:
    /data/o_nejati/Medtsllm/mit-bih/
        100.dat
        100.hea
        100.atr
        RECORDS
        ...

Output folder expected by med-ts-llm ECG anomaly dataset:
    data/mit_ecg/v2/anom/
        train.csv
        test.csv
        test_label.csv
        train_data_desc.csv
        test_data_desc.csv

Important:
    train.csv and test.csv must NOT contain label.
    They must contain only:
        time, patient_id, lead_0, lead_1

    test_label.csv must contain:
        time, patient_id, label
"""


NORMAL_SYMBOLS = {"N"}

NON_BEAT_SYMBOLS = {
    "+", "~", "|", '"', "x", "!", "[", "]",
    "(", ")", "p", "t", "u", "`", "'", "^", "@"
}


def read_records(records_file: Path) -> list[str]:
    with open(records_file, "r") as f:
        records = [line.strip() for line in f if line.strip()]
    return records


def split_records(records: list[str]) -> tuple[list[str], list[str]]:
    """
    Patient-level train/test split.

    Training uses a fixed subset of records.
    Testing uses all remaining records.

    This avoids mixing the same patient's signal between train and test.
    """
    train_records = [
        "100", "101", "103", "105", "106",
        "108", "109", "111", "112", "113",
        "114", "115", "116", "117", "118",
        "119", "121", "122", "123", "124",
    ]

    train_records = [r for r in train_records if r in records]
    test_records = [r for r in records if r not in train_records]

    if len(train_records) == 0:
        raise RuntimeError("No training records found. Check RECORDS file and raw_dir.")

    if len(test_records) == 0:
        raise RuntimeError("No test records found. Check RECORDS file and split_records().")

    return train_records, test_records


def make_point_labels(
    n_samples: int,
    ann_samples: np.ndarray,
    ann_symbols: Iterable[str],
    radius: int,
) -> np.ndarray:
    """
    Convert MIT-BIH beat annotations into dense binary anomaly labels.

    label = 0 means normal.
    label = 1 means anomaly.

    Every abnormal beat annotation marks a small window around that beat
    as anomalous. The window radius is controlled by anomaly_radius.
    """
    labels = np.zeros(n_samples, dtype=np.int32)

    for sample, symbol in zip(ann_samples, ann_symbols):
        if symbol in NON_BEAT_SYMBOLS:
            continue

        if symbol not in NORMAL_SYMBOLS:
            start = max(0, int(sample) - radius)
            end = min(n_samples, int(sample) + radius + 1)
            labels[start:end] = 1

    return labels


def force_two_channels(signal: np.ndarray) -> np.ndarray:
    """
    Force every record to have exactly two signal channels.

    MIT-BIH records can have different lead names, for example:
        MLII, V1
        MLII, V2
        V5, MLII

    We intentionally ignore original lead names and rename the first two
    channels to:
        lead_0, lead_1

    This prevents StandardScaler mismatch:
        X has 4 features, but StandardScaler is expecting 5 features.
    """
    if signal.ndim == 1:
        signal = signal[:, None]

    if signal.shape[1] == 1:
        signal = np.concatenate([signal, signal], axis=1)

    if signal.shape[1] > 2:
        signal = signal[:, :2]

    return signal.astype(np.float32)


def process_record(
    raw_dir: Path,
    record_name: str,
    patient_id: int,
    anomaly_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Process one MIT-BIH WFDB record.

    Returns:
        data_df:
            time, patient_id, lead_0, lead_1

        label_df:
            time, patient_id, label

        desc:
            metadata row for data_desc CSV
    """
    record_path = raw_dir / record_name

    rec = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")

    signal = force_two_channels(rec.p_signal)

    n_samples = signal.shape[0]

    data_df = pd.DataFrame(
        {
            "time": np.arange(n_samples, dtype=np.int64),
            "patient_id": patient_id,
            "lead_0": signal[:, 0],
            "lead_1": signal[:, 1],
        }
    )

    labels = make_point_labels(
        n_samples=n_samples,
        ann_samples=np.asarray(ann.sample),
        ann_symbols=ann.symbol,
        radius=anomaly_radius,
    )

    label_df = pd.DataFrame(
        {
            "time": np.arange(n_samples, dtype=np.int64),
            "patient_id": patient_id,
            "label": labels.astype(np.int32),
        }
    )

    original_leads = rec.sig_name if rec.sig_name is not None else []
    anomaly_ratio = float(label_df["label"].mean())

    desc = {
        "patient_id": patient_id,
        "record": record_name,
        "data_desc": (
            f"MIT-BIH record {record_name}. "
            f"Original leads: {original_leads}. "
            f"Signals are mapped to fixed columns lead_0 and lead_1. "
            f"Sampling frequency: {rec.fs} Hz. "
            f"Binary anomaly labels are derived from non-normal beat annotations. "
            f"Anomaly ratio: {anomaly_ratio:.6f}."
        ),
    }

    return data_df, label_df, desc


def build_split(
    raw_dir: Path,
    records: list[str],
    anomaly_radius: int,
    start_patient_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_parts: list[pd.DataFrame] = []
    label_parts: list[pd.DataFrame] = []
    desc_rows: list[dict] = []

    for i, record_name in enumerate(records):
        patient_id = start_patient_id + i
        print(f"Processing record {record_name} as patient_id={patient_id}")

        data_df, label_df, desc = process_record(
            raw_dir=raw_dir,
            record_name=record_name,
            patient_id=patient_id,
            anomaly_radius=anomaly_radius,
        )

        data_parts.append(data_df)
        label_parts.append(label_df)
        desc_rows.append(desc)

    data = pd.concat(data_parts, axis=0, ignore_index=True)
    labels = pd.concat(label_parts, axis=0, ignore_index=True)

    desc_df = pd.DataFrame(desc_rows)
    desc_df = desc_df.set_index("patient_id")

    return data, labels, desc_df


def final_safety_check(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    test_labels: pd.DataFrame,
) -> None:
    """
    Ensure output CSVs match what datasets/ecg.py expects.

    train.csv:
        time, patient_id, lead_0, lead_1

    test.csv:
        time, patient_id, lead_0, lead_1

    test_label.csv:
        time, patient_id, label
    """
    expected_data_cols = ["time", "patient_id", "lead_0", "lead_1"]
    expected_label_cols = ["time", "patient_id", "label"]

    if train_data.columns.tolist() != expected_data_cols:
        raise ValueError(
            f"Bad train columns: {train_data.columns.tolist()} "
            f"expected {expected_data_cols}"
        )

    if test_data.columns.tolist() != expected_data_cols:
        raise ValueError(
            f"Bad test columns: {test_data.columns.tolist()} "
            f"expected {expected_data_cols}"
        )

    if test_labels.columns.tolist() != expected_label_cols:
        raise ValueError(
            f"Bad test_label columns: {test_labels.columns.tolist()} "
            f"expected {expected_label_cols}"
        )

    if train_data.shape[1] != test_data.shape[1]:
        raise ValueError(
            f"Train/test feature mismatch: "
            f"train has {train_data.shape[1]} columns, "
            f"test has {test_data.shape[1]} columns."
        )

    if "label" in train_data.columns:
        raise ValueError("train.csv must not contain label column.")

    if "label" in test_data.columns:
        raise ValueError("test.csv must not contain label column.")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Path to MIT-BIH folder containing .dat, .hea, .atr, and RECORDS.",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/mit_ecg/v2/anom"),
        help="Output directory expected by MedTsLLM ECG anomaly dataset.",
    )

    parser.add_argument(
        "--anomaly_radius",
        type=int,
        default=18,
        help=(
            "Number of samples around each abnormal beat to mark as anomaly. "
            "At 360 Hz, radius 18 is about 50 ms."
        ),
    )

    args = parser.parse_args()

    records_file = args.raw_dir / "RECORDS"

    if not records_file.exists():
        raise FileNotFoundError(
            f"Could not find RECORDS file at {records_file}"
        )

    records = read_records(records_file)
    train_records, test_records = split_records(records)

    print("\nTrain records:")
    print(train_records)

    print("\nTest records:")
    print(test_records)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_data, train_labels, train_desc = build_split(
        raw_dir=args.raw_dir,
        records=train_records,
        anomaly_radius=args.anomaly_radius,
        start_patient_id=0,
    )

    test_data, test_labels, test_desc = build_split(
        raw_dir=args.raw_dir,
        records=test_records,
        anomaly_radius=args.anomaly_radius,
        start_patient_id=len(train_records),
    )

    # Important:
    # MedTsLLM ECG anomaly dataset expects train.csv/test.csv to have data only.
    # Do not save labels inside train.csv or test.csv.
    train_data = train_data[["time", "patient_id", "lead_0", "lead_1"]].copy()
    test_data = test_data[["time", "patient_id", "lead_0", "lead_1"]].copy()
    test_labels = test_labels[["time", "patient_id", "label"]].copy()

    train_data = train_data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_data = test_data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_labels["label"] = test_labels["label"].astype(np.int32)

    final_safety_check(
        train_data=train_data,
        test_data=test_data,
        test_labels=test_labels,
    )

    train_data.to_csv(args.out_dir / "train.csv", index=False)
    test_data.to_csv(args.out_dir / "test.csv", index=False)
    test_labels.to_csv(args.out_dir / "test_label.csv", index=False)

    train_desc.to_csv(args.out_dir / "train_data_desc.csv")
    test_desc.to_csv(args.out_dir / "test_data_desc.csv")

    print("\nDone.")
    print(f"Output directory: {args.out_dir}")
    print(f"train.csv shape:      {train_data.shape}")
    print(f"test.csv shape:       {test_data.shape}")
    print(f"test_label.csv shape: {test_labels.shape}")
    print(f"Test anomaly ratio:   {test_labels['label'].mean():.6f}")

    print("\nFinal columns:")
    print("train.csv:      ", train_data.columns.tolist())
    print("test.csv:       ", test_data.columns.tolist())
    print("test_label.csv: ", test_labels.columns.tolist())


if __name__ == "__main__":
    main()
