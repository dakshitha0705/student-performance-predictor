"""Download the UCI Student Performance dataset and keep ONLY student-mat.csv.

Run (from the repository root):  python scripts/download_data.py

Writes:
  data/raw/student-mat.csv   (unchanged copy of the source file)
  data/manifest.json         (file name, SHA-256, row count, download date, source URL)

If the download fails (network policy, firewall), download the zip manually from
https://archive.ics.uci.edu/dataset/320/student+performance , extract student-mat.csv
into data/raw/, then run:  python scripts/download_data.py --manifest-only
"""
import io
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA_DIR, RAW_CSV, sha256_file, write_json  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
PAGE = "https://archive.ics.uci.edu/dataset/320/student+performance"


def download() -> None:
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    names = archive.namelist()
    # The outer zip may contain student.zip (nested) or the CSVs directly.
    if "student-mat.csv" in names:
        data = archive.read("student-mat.csv")
    else:
        inner_name = next((n for n in names if n.lower().endswith(".zip")), None)
        if inner_name is None:
            raise SystemExit(f"student-mat.csv not found in archive. Contents: {names}")
        inner = zipfile.ZipFile(io.BytesIO(archive.read(inner_name)))
        data = inner.read("student-mat.csv")
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    RAW_CSV.write_bytes(data)  # byte-for-byte, no re-encoding


def write_manifest() -> None:
    frame = pd.read_csv(RAW_CSV, sep=";")
    manifest = {
        "file_name": RAW_CSV.name,
        "source_url": PAGE,
        "download_url": URL,
        "sha256": sha256_file(RAW_CSV),
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "download_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    if "--manifest-only" not in sys.argv:
        download()
    if not RAW_CSV.exists():
        raise SystemExit(f"{RAW_CSV} not found.")
    write_manifest()
    print("OK: data/raw/student-mat.csv and data/manifest.json are ready.")
