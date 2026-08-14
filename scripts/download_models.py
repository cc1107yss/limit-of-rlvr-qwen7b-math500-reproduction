#!/usr/bin/env python3
"""Download the two public checkpoints at immutable revisions and manifest them."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(
    os.environ.get(
        "REPRO_ROOT",
        "/home/ai/reproductions/limit-of-RLVR-qwen7b-math500-v5",
    )
)
CACHE = ROOT / "hf-cache"
MANIFEST_PATH = ROOT / "manifests" / "models.json"
MODELS = {
    "base": {
        "repo_id": "Qwen/Qwen2.5-7B",
        "revision": "d149729398750b98c0af14eb82c78cfe92750796",
    },
    "simplerl": {
        "repo_id": "hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo",
        "revision": "d630142f26acc8adf8051298cba8023232169d56",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(snapshot: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        files.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256(resolved),
                "cache_blob": resolved.name,
            }
        )
    return files


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": str(CACHE),
        "models": {},
    }
    for label, spec in MODELS.items():
        print(f"DOWNLOAD_START label={label} repo={spec['repo_id']} revision={spec['revision']}", flush=True)
        last_error: Exception | None = None
        for attempt in range(1, 13):
            try:
                snapshot = Path(
                    snapshot_download(
                        repo_id=spec["repo_id"],
                        revision=spec["revision"],
                        cache_dir=CACHE,
                        local_files_only=False,
                        max_workers=1,
                    )
                ).resolve()
                break
            except Exception as error:  # Network transport is deliberately retried; revision never changes.
                last_error = error
                print(
                    f"DOWNLOAD_RETRY label={label} attempt={attempt}/12 "
                    f"error={type(error).__name__}: {error}",
                    flush=True,
                )
                if attempt == 12:
                    raise
                time.sleep(min(120, attempt * 10))
        else:
            raise RuntimeError(f"Unreachable retry state for {label}") from last_error
        if snapshot.name != spec["revision"]:
            raise RuntimeError(f"Resolved revision mismatch for {label}: {snapshot}")
        files = file_manifest(snapshot)
        manifest["models"][label] = {
            **spec,
            "snapshot_path": str(snapshot),
            "file_count": len(files),
            "total_bytes": sum(int(item["size_bytes"]) for item in files),
            "files": files,
        }
        tmp = MANIFEST_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(MANIFEST_PATH)
        print(f"DOWNLOAD_COMPLETE label={label} snapshot={snapshot}", flush=True)
    print(f"MODEL_MANIFEST={MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
