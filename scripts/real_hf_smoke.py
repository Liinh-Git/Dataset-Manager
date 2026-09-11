"""Real Hugging Face storage backend smoke test.

Performs live authentication, write permission validation, synthetic artifact
publication, hash-verified remote loading, and precise single-build cleanup.

If HF_TOKEN or HF_REPO_ID is not set, prints HF-REMOTE-SMOKE: BLOCKED_EXTERNAL
and exits cleanly (exit code 0).
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi

from dataset_manager.config import DatasetBuildConfig
from dataset_manager.hashing import sha256_bytes
from dataset_manager.preprocessing import Samples
from dataset_manager.service import (
    _BuildRecord,
    serialize_durable_record,
)
from dataset_manager.storage import HuggingFaceArtifactStore


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    branch = os.environ.get("HF_BRANCH", "main")

    if not token or not repo_id:
        print("HF-REMOTE-SMOKE: BLOCKED_EXTERNAL (HF_TOKEN or HF_REPO_ID not set in environment)")
        return 0

    print(
        f"[*] Starting Hugging Face real smoke test against repo: '{repo_id}' (branch: '{branch}')"
    )
    api = HfApi(token=token)

    # 1. Non-mutating write permission probe
    try:
        api.auth_check(repo_id=repo_id, repo_type="dataset", token=token, write=True)
        print("[+] Write permission confirmed via auth_check(write=True)")
    except Exception as exc:
        print(f"[-] Write check failed: {type(exc).__name__}: {exc}")
        return 1

    # 2. Branch existence check
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset", token=token)
        branch_names = [b.name for b in refs.branches]
        if branch not in branch_names:
            print(
                f"[-] Branch '{branch}' does not exist in '{repo_id}' (available: {branch_names})"
            )
            return 1
        print(f"[+] Verified branch '{branch}' exists on remote")
    except Exception as exc:
        print(f"[-] Branch validation failed: {type(exc).__name__}: {exc}")
        return 1

    smoke_build_id = f"smoke-{uuid.uuid4().hex}"
    print(f"[*] Generated smoke build ID: {smoke_build_id}")
    cleanup_successful = True

    with tempfile.TemporaryDirectory() as temp_root:
        local_store = Path(temp_root) / "store"
        store = HuggingFaceArtifactStore(
            local_root=local_store,
            repo_id=repo_id,
            token=token,
            branch=branch,
            api=api,
        )

        build_config = DatasetBuildConfig(
            1,
            smoke_build_id,
            "cifar10",
            "CIFAR-10",
            "cifar10_smoke",
            "image_classification",
            (3, 2, 2),
            "float32",
            10,
            {
                "channel_order": "NCHW",
                "scale": "uint8_to_unit",
                "mean": (0.5, 0.5, 0.5),
                "std": (0.5, 0.5, 0.5),
            },
            2,
            2,
            "seeded_permutation_round_robin",
            42,
        )

        # Generate synthetic 4-sample dataset
        x = np.zeros((4, 3, 2, 2), dtype=np.float32)
        y = np.arange(4, dtype=np.int64) % 10
        sample_ids = np.arange(4, dtype=np.int64)
        samples = Samples(x=x, y=y, sample_ids=sample_ids)

        try:
            # 3. Materialize and upload to remote HF repo
            print("[*] Materializing synthetic artifacts and uploading to Hugging Face...")
            published = store.materialize(build_config, samples)
            manifest_hash = published.dataset_manifest_hash
            print(f"[+] Materialized and verified remote hash: {manifest_hash}")

            # 4. Upload durable metadata record
            record = _BuildRecord(
                dataset_build_id=smoke_build_id,
                request={"smoke": True},
                request_fingerprint="smoke-fp",
                idempotency_key_hash=sha256_bytes(b"smoke-key"),
                state="READY",
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                dataset_manifest_hash=manifest_hash,
                sample_count=4,
            )
            record_data = serialize_durable_record(record, is_remote=True)
            store.persist_record(record_data)
            print("[+] Persisted durable metadata record to remote Hugging Face repo")

            # 5. Cold-start verification from fresh directory
            fresh_cache = Path(temp_root) / "fresh_cache"
            fresh_store = HuggingFaceArtifactStore(
                local_root=fresh_cache,
                repo_id=repo_id,
                token=token,
                branch=branch,
                api=api,
            )

            # Verify loading metadata records
            records = fresh_store.load_records()
            found = [r for r in records if r.get("dataset_build_id") == smoke_build_id]
            if not found:
                print(f"[-] Could not find persisted metadata for {smoke_build_id} in remote repo")
                return 1
            print("[+] Successfully verified remote metadata record retrieval")

            # Verify downloading root manifest and batch NPZ on cache miss
            loaded = fresh_store.load(smoke_build_id, expected_manifest_hash=manifest_hash)
            batch_file = fresh_store.resolve_artifact(loaded, "shards/000/batch-000000.npz")
            if not batch_file.is_file():
                print("[-] Failed to download batch artifact on cache miss")
                return 1
            print("[+] Successfully resolved and verified batch NPZ on cache miss")

        finally:
            # 6. Strict precise cleanup: remove ONLY this build's artifact folder and record
            print("[*] Cleaning up smoke test artifacts from remote Hugging Face repo...")
            try:
                api.delete_folder(
                    path_in_repo=f"dataset-builds/{smoke_build_id}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=branch,
                    token=token,
                    commit_message=f"Clean up smoke artifact folder {smoke_build_id}",
                )
                print(f"[+] Deleted remote folder dataset-builds/{smoke_build_id}")
            except Exception as exc:
                cleanup_successful = False
                print(f"[-] ERROR: Failed to delete remote folder {smoke_build_id}: {exc}")

            try:
                api.delete_file(
                    path_in_repo=f".service/builds/{smoke_build_id}.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=branch,
                    token=token,
                    commit_message=f"Clean up smoke metadata record {smoke_build_id}",
                )
                print(f"[+] Deleted remote file .service/builds/{smoke_build_id}.json")
            except Exception as exc:
                cleanup_successful = False
                print(f"[-] ERROR: Failed to delete metadata {smoke_build_id}: {exc}")

    if not cleanup_successful:
        print("[-] HF-REMOTE-SMOKE: FAIL (Artifact or metadata cleanup failed)")
        return 1

    print("HF-REMOTE-SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
