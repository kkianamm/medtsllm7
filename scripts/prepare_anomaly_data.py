# scripts/prepare_mitbih_anomaly.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import wfdb


# MIT-BIH beat annotation symbols.
# Normal beats are treated as non-anomaly.
# Everything else is anomaly for binary anomaly detection.
NORMAL_SYMBOLS = {"N"}

# Common non-beat symbols that should not create anomaly labels.
# They mark rhythm changes, comments, signal quality, etc.
NON_BEAT_SYMBOLS = {
    "+", "~", "|", '"', "x", "!", "[", "]", "(", ")", "p", "t", "u", "`", "'", "^", "@"
}


def read_records(records_file: Path) -> list[str]:
    with open(records_file, "r") as f:
        records = [line.strip() for line in f if line.strip()]
    return records


def split_records(records: list[str]) -> tuple[list[str], list[str]]:
    """
    Patient-level split.

    Train uses mostly normal records.
    Test includes more arrhythmia-heavy records.

    You can change this later for experiments.
    """
    train_records = [
        "100", "101", "103", "105", "106", "108", "109", "111",
        "112", "113", "114", "115", "116", "117", "118", "119",
        "121", "122", "123", "124",
    ]

    train_records = [r for r in train_records if r in records]
    test_records = [r for r in records if r not in train_records]

    return train_records, test_records


def make_point_labels(
    n_samples: int,
    ann_samples: np.ndarray,
    ann_symbols: Iterable[str],
    radius: int,
) -> np.ndarray:
    """
    Convert beat annotations to dense point-level anomaly labels.

    label = 0: normal
    label = 1: anomalous beat region

    A small radius around each abnormal beat is marked as anomaly.
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


def process_record(
    raw_dir: Path,
    record_name: str,
    patient_id: int,
    anomaly_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Output data columns:
      time, patient_id, MLII, V1

    Output label columns:
      time, patient_id, label
    """
    record_path = raw_dir / record_name

    rec = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")

    signal = rec.p_signal.astype(np.float32)

    if signal.ndim == 1:
        signal = signal[:, None]

    if signal.shape[1] == 1:
        sig_df = pd.DataFrame(signal, columns=["lead_0"])
    else:
        # MIT-BIH usually has two channels.
        # Use header channel names when available.
        names = rec.sig_name if rec.sig_name is not None else ["lead_0", "lead_1"]
        names = [str(x).replace(" ", "_") for x in names]
        sig_df = pd.DataFrame(signal, columns=names[: signal.shape[1]])

    n_samples = len(sig_df)
    labels = make_point_labels(
        n_samples=n_samples,
        ann_samples=np.asarray(ann.sample),
        ann_symbols=ann.symbol,
        radius=anomaly_radius,
    )

    sig_df.insert(0, "patient_id", patient_id)
    sig_df.insert(0, "time", np.arange(n_samples, dtype=np.int64))

    label_df = pd.DataFrame(
        {
            "time": np.arange(n_samples, dtype=np.int64),
            "patient_id": patient_id,
            "label": labels,
        }
    )

    return sig_df, label_df


def build_split(
    raw_dir: Path,
    records: list[str],
    anomaly_radius: int,
    start_patient_id: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_parts = []
    label_parts = []
    desc_rows = []

    for i, record_name in enumerate(records):
        patient_id = start_patient_id + i
        print(f"Processing record {record_name} as patient_id={patient_id}")

        data_df, label_df = process_record(
            raw_dir=raw_dir,
            record_name=record_name,
            patient_id=patient_id,
            anomaly_radius=anomaly_radius,
        )

        data_parts.append(data_df)
        label_parts.append(label_df)

        anomaly_rate = float(label_df["label"].mean())
        desc_rows.append(
            {
                "patient_id": patient_id,
                "data_desc": (
                    f"MIT-BIH record {record_name}. "
                    f"Two-channel ambulatory ECG sampled at 360 Hz. "
                    f"Binary anomaly labels are derived from abnormal beat annotations. "
                    f"Anomaly ratio: {anomaly_rate:.4f}."
                ),
            }
        )

    data = pd.concat(data_parts, axis=0, ignore_index=True)
    labels = pd.concat(label_parts, axis=0, ignore_index=True)
    desc = pd.DataFrame(desc_rows).set_index("patient_id")

    return data, labels, desc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Path to MIT-BIH WFDB folder containing .dat/.hea/.atr files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/mit_ecg/v2/anom"),
        help="Output folder expected by datasets/ecg.py.",
    )
    parser.add_argument(
        "--anomaly_radius",
        type=int,
        default=18,
        help=(
            "Samples around abnormal beat annotation to mark as anomaly. "
            "18 samples is about 50 ms at 360 Hz."
        ),
    )

    args = parser.parse_args()

    records_file = args.raw_dir / "RECORDS"
    if not records_file.exists():
        raise FileNotFoundError(f"Missing RECORDS file: {records_file}")

    records = read_records(records_file)
    train_records, test_records = split_records(records)

    print("Train records:", train_records)
    print("Test records:", test_records)

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

    # MedTsLLM ECG anomaly class expects train.csv and test.csv without label column.
    train_data.to_csv(args.out_dir / "train.csv", index=False)
    test_data.to_csv(args.out_dir / "test.csv", index=False)

    # It only reads test_label.csv for non-train splits.
    test_labels.to_csv(args.out_dir / "test_label.csv", index=False)

    train_desc.to_csv(args.out_dir / "train_data_desc.csv")
    test_desc.to_csv(args.out_dir / "test_data_desc.csv")

    print(f"\nDone. Wrote files to: {args.out_dir}")
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape:  {test_data.shape}")
    print(f"Test anomaly ratio: {test_labels['label'].mean():.4f}")


if __name__ == "__main__":
    main()
