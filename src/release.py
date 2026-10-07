"""Controlled model switching and rollback.

  python -m src.release status             show which model the manifest points to
  python -m src.release backup             copy active-model.json -> active-model.backup.json
  python -m src.release activate v2        point active-model.json at artifacts/model-v2.joblib
  python -m src.release restore            put the backed-up manifest back

After ANY change, restart the API (it loads the model once at startup):
  local:   stop uvicorn and start it again
  docker:  docker compose restart api
"""
from __future__ import annotations

import shutil
import sys

from .common import ACTIVE_MODEL_PATH, ARTIFACTS_DIR, read_json, write_json

BACKUP_PATH = ARTIFACTS_DIR / "active-model.backup.json"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    command = argv[0]

    if command == "status":
        print(read_json(ACTIVE_MODEL_PATH))
        return 0

    if command == "backup":
        shutil.copyfile(ACTIVE_MODEL_PATH, BACKUP_PATH)
        print(f"Backed up {ACTIVE_MODEL_PATH.name} -> {BACKUP_PATH.name}: {read_json(BACKUP_PATH)}")
        return 0

    if command == "activate":
        if len(argv) < 2:
            print("Usage: python -m src.release activate v2")
            return 1
        version = argv[1]
        artifact = ARTIFACTS_DIR / f"model-{version}.joblib"
        if not artifact.exists() or not (ARTIFACTS_DIR / f"model-{version}.json").exists():
            print(f"ERROR: artifact or metadata for {version} not found in artifacts/")
            return 1
        if not BACKUP_PATH.exists():
            print("ERROR: run `python -m src.release backup` first so you can roll back.")
            return 1
        write_json(ACTIVE_MODEL_PATH, {"model_version": version, "artifact_path": f"artifacts/model-{version}.joblib"})
        print(f"Active model is now {version}. Restart the API to load it.")
        return 0

    if command == "restore":
        if not BACKUP_PATH.exists():
            print("ERROR: no backup found.")
            return 1
        shutil.copyfile(BACKUP_PATH, ACTIVE_MODEL_PATH)
        print(f"Restored: {read_json(ACTIVE_MODEL_PATH)}. Restart the API to load it.")
        return 0

    print(f"Unknown command: {command}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
