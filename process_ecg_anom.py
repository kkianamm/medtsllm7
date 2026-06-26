"""
process_ecg_anom.py  --  MIT-BIH preprocessing for MedTsLLM anomaly detection.

Implements the paper's MIT-BIH protocol (MLHC 2024):
  * 48 records / 47 patients; keep ONLY records that contain BOTH MLII and V1 leads.
  * downsample 360 Hz -> 125 Hz.
  * beats classified normal vs abnormal from expert annotations (AAMI EC57 mapping).
  * abnormal annotations expanded to 300 ms windows (+/-150 ms).
  * 80/20 patient split, FAVORING minimal-anomaly patients for training
    (reconstruction-style training on mostly-normal signal; abnormal-heavy patients -> test).

Produces exactly the files datasets/ecg.py :: ECGMITAnomalyDetectionDataset reads:
  data/mit_ecg/v2/anom/{train,test}.csv          time, patient_id, MLII, V1
  data/mit_ecg/v2/anom/test_label.csv            time, patient_id, label   (row-aligned to test.csv)
  data/mit_ecg/v2/anom/{train,test}_data_desc.csv  index: patient_id, column: data_desc

Usage:
  python3 datasets/process_ecg_anom.py --selftest        # offline unit tests, no download
  python3 datasets/process_ecg_anom.py --synthetic       # fabricate data, no download
  python3 datasets/process_ecg_anom.py                   # real MIT-BIH (needs: pip install wfdb scipy)
  python3 datasets/process_ecg_anom.py --minutes 5       # cap each record to 5 min (faster/smaller)
"""

import argparse
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

OUT_DIR   = Path(__file__).parent / "../data/mit_ecg/v2/anom"
FS_TARGET = 125
WIN       = 128                       # MUST match history_len == pred_len in the config
LEADS     = ("MLII", "V1")            # paper: keep only records with BOTH of these
ANOM_MS   = 150                       # +/-150 ms  ->  300 ms total window
TRAIN_RATIO = 0.8

# 48 MIT-BIH Arrhythmia Database records
ALL_RECORDS = [
    "100","101","102","103","104","105","106","107","108","109","111","112",
    "113","114","115","116","117","118","119","121","122","123","124","200",
    "201","202","203","205","207","208","209","210","212","213","214","215",
    "217","219","220","221","222","223","228","230","231","232","233","234",
]

# AAMI EC57 beat-class mapping.  Anything not a beat symbol (rhythm/quality marks) is ignored.
AAMI_NORMAL   = set("NLRej")                 # class N
AAMI_ABNORMAL = set("AaJS") | set("VE") | set("F") | set("/fQ")   # S, V, F, Q classes


# ---------------------------------------------------------------- pure functions (unit-tested) #
def resample_signal(sig: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Polyphase resample along time axis (axis 0). sig: [n, channels]."""
    if fs_in == fs_out:
        return np.asarray(sig, dtype=np.float64)
    g = gcd(fs_out, fs_in)
    return resample_poly(np.asarray(sig, dtype=np.float64), fs_out // g, fs_in // g, axis=0)


def ann_to_labels(samples, symbols, n_in, n_out, fs_out, anom_ms=ANOM_MS):
    """Map annotation sample indices (at fs_in resolution -> n_in length) onto a
    length-n_out label vector, expanding ABNORMAL beats to +/-anom_ms windows."""
    labels = np.zeros(n_out, dtype=np.int64)
    half = round(anom_ms / 1000.0 * fs_out)
    scale = n_out / n_in
    for s, sym in zip(samples, symbols):
        if sym not in AAMI_ABNORMAL:        # normal beats and non-beat marks -> no anomaly
            continue
        c = int(round(s * scale))
        labels[max(0, c - half): min(n_out, c + half + 1)] = 1
    return labels


def anomaly_fraction(labels: np.ndarray) -> float:
    return float(np.mean(labels)) if len(labels) else 0.0


def split_by_anomaly(frac_by_pid: dict, train_ratio=TRAIN_RATIO):
    """80/20 patient split; the LOW-anomaly patients go to train."""
    order = sorted(frac_by_pid, key=lambda p: frac_by_pid[p])   # ascending anomaly fraction
    n_train = max(1, round(train_ratio * len(order)))
    n_train = min(n_train, len(order) - 1)                      # guarantee >=1 test patient
    return order[:n_train], order[n_train:]


def trim_to_window(n_rows: int, win=WIN) -> int:
    return (n_rows // win) * win


# --------------------------------------------------------------------------------- IO helpers #
def _patient_frame(pid, sig_2ch, labels):
    n = trim_to_window(sig_2ch.shape[0])
    df = pd.DataFrame(sig_2ch[:n], columns=list(LEADS))
    df.insert(0, "patient_id", int(pid))
    df.insert(0, "time", np.arange(n))
    df["label"] = labels[:n]
    return df


def _write_split(frames, split, with_labels):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.concat(frames, ignore_index=True)

    feat_cols = list(LEADS)
    rows[["time", "patient_id"] + feat_cols].to_csv(OUT_DIR / f"{split}.csv", index=False)
    if with_labels:
        rows[["time", "patient_id", "label"]].to_csv(OUT_DIR / f"{split}_label.csv", index=False)

    descs = {
        int(pid): f"Patient {int(pid)}: MIT-BIH ambulatory ECG, leads MLII/V1, "
                  f"downsampled to {FS_TARGET} Hz, AAMI beat annotations."
        for pid in rows["patient_id"].unique()
    }
    pd.DataFrame({"patient_id": list(descs), "data_desc": list(descs.values())}) \
        .set_index("patient_id").to_csv(OUT_DIR / f"{split}_data_desc.csv")

    print(f"  {split}: {len(rows):,} rows | {rows['patient_id'].nunique()} patients | "
          f"anomaly rate {anomaly_fraction(rows['label'].values):.4f}")
    return rows


# ----------------------------------------------------------------------------------- real mode #
def load_records(records, minutes=None):
    """Returns {pid: (sig_2ch[n,2] @125Hz, labels[n], anomaly_fraction)} for eligible records."""
    import wfdb
    out = {}
    for rec_id in records:
        try:
            rec = wfdb.rdrecord(rec_id, pn_dir="mitdb")
            ann = wfdb.rdann(rec_id, "atr", pn_dir="mitdb")
        except Exception as e:
            print(f"  skip {rec_id}: download/read failed ({e})")
            continue

        names = list(rec.sig_name)
        if not all(l in names for l in LEADS):       # paper: require BOTH MLII and V1
            print(f"  skip {rec_id}: leads {names} lack {LEADS}")
            continue

        idx = [names.index(LEADS[0]), names.index(LEADS[1])]
        sig = rec.p_signal[:, idx]
        if minutes is not None:
            sig = sig[: int(minutes * 60 * rec.fs)]

        n_in = sig.shape[0]
        sig125 = resample_signal(sig, rec.fs, FS_TARGET)
        labels = ann_to_labels(ann.sample, ann.symbol, n_in, sig125.shape[0], FS_TARGET)
        out[int(rec_id)] = (sig125, labels, anomaly_fraction(labels))
        print(f"  {rec_id}: {n_in}@{rec.fs}Hz -> {sig125.shape[0]}@{FS_TARGET}Hz | "
              f"anomaly {out[int(rec_id)][2]:.4f}")
    return out


def make_real(records, minutes):
    print(f"Loading MIT-BIH (pn_dir='mitdb'), require leads {LEADS}, downsample to {FS_TARGET} Hz...")
    recs = load_records(records, minutes)
    if len(recs) < 2:
        raise RuntimeError(f"Only {len(recs)} eligible record(s); need >=2. Check wfdb/network.")

    frac = {pid: recs[pid][2] for pid in recs}
    train_ids, test_ids = split_by_anomaly(frac)
    print(f"\nSplit (low-anomaly -> train): train={sorted(train_ids)}  test={sorted(test_ids)}")

    _write_split([_patient_frame(p, recs[p][0], recs[p][1]) for p in train_ids], "train", False)
    _write_split([_patient_frame(p, recs[p][0], recs[p][1]) for p in test_ids],  "test",  True)


# -------------------------------------------------------------------------------- synthetic mode #
def make_synthetic(seed=0):
    rng = np.random.default_rng(seed)

    def gen(pid, secs, anom):
        n = secs * FS_TARGET
        t = np.arange(n)
        s0 = 0.6 * np.sin(2*np.pi*t/40) + 0.15*rng.standard_normal(n) + ((t % 40) < 3) * 1.4
        s1 = 0.4 * np.sin(2*np.pi*t/40 + 0.7) + 0.12*rng.standard_normal(n)
        lab = np.zeros(n, dtype=np.int64)
        if anom:
            for _ in range(max(1, n // 1500)):
                a = rng.integers(0, n - 60); b = a + rng.integers(20, 50)
                s0[a:b] += 2.5*rng.standard_normal(b-a); s1[a:b] += 2.0*rng.standard_normal(b-a)
                lab[a:b] = 1
        sig = np.stack([s0, s1], axis=1)
        return _patient_frame(pid, sig, lab)

    print("Synthetic ECG (MLII/V1-style):")
    _write_split([gen(p, 60, False) for p in (100, 101, 103)], "train", False)
    _write_split([gen(p, 40, True)  for p in (200, 201)],      "test",  True)


# --------------------------------------------------------------------------------------- tests #
def selftest():
    print("Running offline self-tests...")

    # resampling length ~ ratio
    x = np.random.randn(3600, 2)
    y = resample_signal(x, 360, 125)
    assert abs(y.shape[0] - 1250) <= 2, y.shape
    assert y.shape[1] == 2

    # label window = 300 ms (+/-150ms => 39 samples incl. center) around an ABNORMAL beat only
    n_in, n_out = 3600, 1250
    labels = ann_to_labels([0, 1800], ["N", "V"], n_in, n_out, FS_TARGET)
    half = round(ANOM_MS/1000*FS_TARGET)
    assert labels[: half].sum() == 0, "normal beat must not be flagged"
    center = round(1800 * n_out / n_in)
    assert labels[center] == 1 and labels[center - half] == 1 and labels[center + half] == 1
    assert labels.sum() == (2*half + 1), labels.sum()

    # split favors low-anomaly patients for train
    tr, te = split_by_anomaly({1: 0.30, 2: 0.01, 3: 0.20, 4: 0.02, 5: 0.40})
    assert set(tr) == {2, 4, 3, 1} and te == [5], (tr, te)   # 80% lowest -> train

    # window trim
    assert trim_to_window(1250) == 1152 and 1152 % WIN == 0

    print("  all self-tests PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--minutes", type=float, default=None, help="cap each record to N minutes")
    ap.add_argument("--records", nargs="*", default=ALL_RECORDS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if args.synthetic:
        make_synthetic(args.seed)
    else:
        make_real(args.records, args.minutes)
    print(f"\nDone -> {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
