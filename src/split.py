"""Stage 2: create the fixed 60/15/15/10 partitions.

Run:  python -m src.split

Procedure (guide section 9), random_state=42, shuffle=True:
  1. Reserve 10% of clean rows as `later` (the "new data" batch for retraining).
  2. From the remaining 90%, reserve one third as the evaluation pool.
  3. Split that pool equally into `validation` and final `test`.
Result is about 60 / 15 / 15 / 10 after integer rounding. This is a random
educational benchmark, not a chronological validation.
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from .common import DATA_DIR, PROCESSED_DIR, SEED, sha256_file, sha256_text, utc_now, write_json

PARTITION_NAMES = ["train", "validation", "test", "later"]


def make_partitions(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rest, later = train_test_split(df, test_size=0.10, random_state=seed, shuffle=True)
    train, pool = train_test_split(rest, test_size=1 / 3, random_state=seed, shuffle=True)
    validation, test = train_test_split(pool, test_size=0.5, random_state=seed, shuffle=True)
    return {"train": train, "validation": validation, "test": test, "later": later}


def ids_hash(ids: list[int]) -> str:
    return sha256_text(",".join(str(i) for i in sorted(ids)))


def main() -> int:
    clean_path = PROCESSED_DIR / "clean.csv"
    if not clean_path.exists():
        print("ERROR: data/processed/clean.csv not found. Run `python -m src.validate` first.")
        return 1

    df = pd.read_csv(clean_path)
    parts = make_partitions(df)

    # Self-check: every row in exactly one partition.
    all_ids = [i for p in parts.values() for i in p["source_row_id"].tolist()]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(df["source_row_id"]):
        print("ERROR: partitions are not disjoint and complete.")
        return 1

    manifest = {
        "created_at_utc": utc_now(),
        "seed": SEED,
        "procedure": "later=10%; from remaining 90%, evaluation pool=1/3; pool split equally into validation/test",
        "clean_csv_sha256": sha256_file(clean_path),
        "total_rows": int(len(df)),
        "partitions": {},
    }
    for name in PARTITION_NAMES:
        part = parts[name].reset_index(drop=True)
        csv_path = PROCESSED_DIR / f"{name}.csv"
        part.to_csv(csv_path, index=False)
        ids = part["source_row_id"].tolist()
        (PROCESSED_DIR / f"{name}_row_ids.txt").write_text(
            "\n".join(str(i) for i in ids) + "\n", encoding="utf-8")
        manifest["partitions"][name] = {
            "rows": int(len(part)),
            "share": round(len(part) / len(df), 4),
            "row_ids_sha256": ids_hash(ids),
            "csv_sha256": sha256_file(csv_path),
        }

    write_json(DATA_DIR / "splits.json", manifest)
    for name in PARTITION_NAMES:
        info = manifest["partitions"][name]
        print(f"{name:<11} {info['rows']:>4} rows ({info['share']:.1%})")
    print("OK: wrote data/processed/*.csv and data/splits.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
