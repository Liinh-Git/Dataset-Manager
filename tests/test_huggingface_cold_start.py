from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataset_manager import storage
from dataset_manager.config import DatasetBuildConfig, DatasetManagerConfig
from dataset_manager.preprocessing import Preprocessor, Samples
from dataset_manager.schemas import DatasetBuildState
from dataset_manager.service import BuildExecutionResult, DatasetService, DatasetServiceError
from test_huggingface_storage import FakeHfApi


def _make_sample_data(count: int = 4) -> Samples:
    x = np.arange(count * 12, dtype=np.uint8).reshape(count, 3, 2, 2)
    y = np.arange(count, dtype=np.int64) % 10
    return Preprocessor((3, 2, 2), 10, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).transform(x, y)


def _make_request() -> dict[str, Any]:
    return {
        "command_id": "cmd-1",
        "dataset_id": "cifar10",
        "profile": "CNN_IMAGE_CLASSIFICATION_V1",
        "input_shape": [3, 2, 2],
        "batch_size": 2,
        "shard_count": 2,
        "partition_seed": 42,
        "normalization": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "source": {
            "type": "cifar10_binary",
            "version": "binary-v1",
        },
    }


def test_huggingface_cold_start_recovery_and_serving(tmp_path: Path):
    """Real lifecycle cold start test:

    submit -> actual build execution -> REGISTERING -> acknowledge_registration -> READY
    -> stop service -> completely empty local directories -> new service instance
    -> recover READY from HF -> same idempotency key resolves same build
    -> root/shard/batch artifacts download and verify cryptographically.
    """
    api = FakeHfApi()
    store1_dir = tmp_path / "store1"
    temp1_dir = tmp_path / "temp1"

    config1 = DatasetManagerConfig(
        storage_backend="huggingface",
        hf_repo_id="test/repo",
        hf_token="secret-token",
        hf_branch="main",
        store_dir=str(store1_dir),
        temp_dir=str(temp1_dir),
    )

    orig_init = storage.HuggingFaceArtifactStore.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["api"] = api
        orig_init(self, *args, **kwargs)

    storage.HuggingFaceArtifactStore.__init__ = patched_init

    try:

        def mock_executor(
            build_id: str,
            request: dict[str, Any],
            update: Any,
        ) -> BuildExecutionResult:
            workspace = Path(config1.temp_dir).resolve() / build_id
            workspace.mkdir(parents=True, exist_ok=True)
            build_config = DatasetBuildConfig(
                1,
                build_id,
                "cifar10",
                "CIFAR-10",
                request["profile"],
                "image_classification",
                tuple(request["input_shape"]),
                "float32",
                10,
                {
                    "channel_order": "NCHW",
                    "scale": "uint8_to_unit",
                    "mean": request["normalization"]["mean"],
                    "std": request["normalization"]["std"],
                },
                request["batch_size"],
                request["shard_count"],
                "seeded_permutation_round_robin",
                request["partition_seed"],
            )
            samples = _make_sample_data(4)
            update(DatasetBuildState.MATERIALIZING, "MATERIALIZING", 0.5)
            published = service1.storage.materialize(build_config, samples)
            update(DatasetBuildState.VERIFYING, "VERIFYING", 0.9)
            return BuildExecutionResult(published=published, raw_workspace=workspace)

        # 1. First instance: start with worker enabled
        service1 = DatasetService(config1, executor=mock_executor, start_worker=True)

        # Submit build
        sub = service1.submit(_make_request(), idempotency_key="cold-start-key-1")
        build_id = str(sub["dataset_build_id"])

        # Wait for actual build worker execution to reach REGISTERING state
        deadline = time.time() + 5.0
        while time.time() < deadline:
            st = service1.status(build_id)
            if st["state"] == DatasetBuildState.REGISTERING.value:
                break
            time.sleep(0.05)
        else:
            pytest.fail(
                f"Build did not reach REGISTERING in time, current: {service1.status(build_id)}"
            )

        reg_status = service1.status(build_id)
        assert reg_status["state"] == "REGISTERING"
        manifest_hash = str(reg_status["dataset_manifest_hash"])
        assert manifest_hash
        expected_origin = (
            f"https://huggingface.co/datasets/test/repo/resolve/main/dataset-builds/{build_id}"
        )
        assert reg_status["artifact_base_url"] == expected_origin
        assert reg_status["manifest_uri"] == f"{expected_origin}/dataset-manifest.json"

        # Complete lifecycle to READY via acknowledge_registration
        ack_res = service1.acknowledge_registration(
            dataset_build_id=build_id,
            dataset_manifest_hash=manifest_hash,
            registration_id="reg-real-lifecycle",
            catalog_persisted_at="2026-01-01T00:00:00Z",
        )
        assert ack_res["state"] == "READY"
        assert ack_res["registration_id"] == "reg-real-lifecycle"

        # Check remote HF files exist
        assert f".service/builds/{build_id}.json" in api.files
        assert f"dataset-builds/{build_id}/dataset-manifest.json" in api.files

        # Remote metadata file must NOT contain raw idempotency_key or raw_workspace
        remote_meta = json.loads(api.files[f".service/builds/{build_id}.json"])
        assert "idempotency_key" not in remote_meta
        assert "raw_workspace" not in remote_meta
        assert remote_meta["idempotency_key_hash"] != ""

        service1.close()

        # 2. SIMULATE CONTAINER TOTAL WIPE:
        # Completely delete all local storage and temp directories
        if store1_dir.exists():
            shutil.rmtree(store1_dir)
        if temp1_dir.exists():
            shutil.rmtree(temp1_dir)

        # Fresh empty local directories for service2
        store2_dir = tmp_path / "store2_fresh"
        temp2_dir = tmp_path / "temp2_fresh"
        config2 = DatasetManagerConfig(
            storage_backend="huggingface",
            hf_repo_id="test/repo",
            hf_token="secret-token",
            hf_branch="main",
            store_dir=str(store2_dir),
            temp_dir=str(temp2_dir),
        )

        service2 = DatasetService(config2, start_worker=False)

        # A. Health check
        h = service2.health()
        assert h["status"] == "ok"
        assert h["storage_writable"] is True

        # B. Status check restored from remote HF
        st = service2.status(build_id)
        assert st["state"] == "READY"
        assert st["dataset_manifest_hash"] == manifest_hash
        assert st["sample_count"] == 4
        assert st["registration_id"] == "reg-real-lifecycle"

        # C. Idempotency test survives cold start
        same_sub = service2.submit(_make_request(), idempotency_key="cold-start-key-1")
        assert same_sub["dataset_build_id"] == build_id

        # D. Conflict test
        diff_req = _make_request()
        diff_req["batch_size"] = 1
        with pytest.raises(DatasetServiceError) as exc_info:
            service2.submit(diff_req, idempotency_key="cold-start-key-1")
        assert exc_info.value.code == "IDEMPOTENCY_CONFLICT"

        # E. Artifact downloading on cache miss survives cold start
        # Root manifest download and cryptographic hash verification
        root_path, root_hash, media = service2.artifact(build_id)
        assert root_hash == manifest_hash
        assert root_path.is_file()
        assert media == "application/json"

        # Shard manifest download and cryptographic hash verification
        shard_path, _shard_hash, media = service2.artifact(build_id, shard_id=0)
        assert shard_path.is_file()
        assert media == "application/json"

        # Batch NPZ download and cryptographic hash/size verification
        batch_path, _batch_hash, media = service2.artifact(build_id, shard_id=0, batch_id=0)
        assert batch_path.is_file()
        assert media == "application/octet-stream"

        # F. Deprecation and Purge
        dep_status = service2.deprecate(build_id)
        assert dep_status["state"] == "DEPRECATED"

        del_status = service2.purge(build_id, command_id="purge-cmd-1")
        assert del_status["state"] == "DELETED"

        # Verify remote artifact folder was deleted
        assert not any(k.startswith(f"dataset-builds/{build_id}/") for k in api.files)

        service2.close()
    finally:
        storage.HuggingFaceArtifactStore.__init__ = orig_init
