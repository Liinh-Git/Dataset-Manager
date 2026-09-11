"""Automated verification of filesystem restart durability and idempotency."""

import shutil
import time
from pathlib import Path

import numpy as np

from dataset_manager.config import DatasetBuildConfig, DatasetManagerConfig
from dataset_manager.preprocessing import Preprocessor
from dataset_manager.service import BuildExecutionResult, DatasetService
from dataset_manager.storage import DatasetStorage


def fake_executor(store_root: Path):
    def _exec(build_id, request, update):
        config = DatasetBuildConfig(
            1,
            build_id,
            "cifar10",
            "CIFAR-10",
            "CNN_IMAGE_CLASSIFICATION_V1",
            "image_classification",
            (3, 2, 2),
            "float32",
            10,
            {
                "channel_order": "NCHW",
                "scale": "uint8_to_unit",
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            2,
            3,
            "seeded_permutation_round_robin",
            42,
        )
        raw = np.arange(6 * 12, dtype=np.uint8).reshape(6, 3, 2, 2)
        samples = Preprocessor((3, 2, 2), 10, (0.5,) * 3, (0.5,) * 3).transform(
            raw, np.arange(6, dtype=np.int64)
        )
        published = DatasetStorage(store_root).materialize(config, samples)
        return BuildExecutionResult(published)

    return _exec


def test_filesystem_restart_durability_and_idempotency(tmp_path: Path):
    store_dir = tmp_path / "durable_store"
    temp_dir = tmp_path / "ephemeral_temp"
    idempotency_key = "idemp-durability-test-123"

    config1 = DatasetManagerConfig(
        host="127.0.0.1",
        port=9200,
        store_dir=str(store_dir),
        temp_dir=str(temp_dir),
        public_base_url="http://127.0.0.1:9200",
    )
    service1 = DatasetService(config1, executor=fake_executor(store_dir))
    try:
        request_body = {
            "command_id": "cmd-restart-test",
            "source": {
                "type": "cifar10_download",
                "dataset_name": "cifar10",
                "version": "binary-v1",
            },
            "profile": "CNN_IMAGE_CLASSIFICATION_V1",
            "input_shape": [3, 32, 32],
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "batch_size": 2,
            "shard_count": 3,
            "partition_seed": 42,
        }
        sub = service1.submit(request_body, idempotency_key)
        build_id = sub["dataset_build_id"]

        start = time.time()
        while time.time() - start < 10:
            st = service1.status(build_id)
            if st["state"] in ("REGISTERING", "READY"):
                break
            time.sleep(0.05)

        manifest_hash = st["dataset_manifest_hash"]
        ack = service1.acknowledge_registration(
            build_id,
            manifest_hash,
            "reg-durability-ack",
            "2026-09-11T12:00:00Z",
        )
        assert ack["state"] == "READY"

        root_path, root_hash, _ = service1.artifact(build_id)
        root_bytes = root_path.read_bytes()
        shard_path, shard_hash, _ = service1.artifact(build_id, shard_id=0)
        shard_bytes = shard_path.read_bytes()
        batch_path, batch_hash, _ = service1.artifact(build_id, shard_id=0, batch_id=0)
        batch_bytes = batch_path.read_bytes()
    finally:
        service1.close()

    # Delete ephemeral temp directory; DO NOT delete durable store
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    assert not temp_dir.exists()
    assert store_dir.exists()

    # Restart DatasetService from same durable store
    config2 = DatasetManagerConfig(
        host="127.0.0.1",
        port=9200,
        store_dir=str(store_dir),
        temp_dir=str(temp_dir),
        public_base_url="http://127.0.0.1:9200",
    )
    service2 = DatasetService(config2, executor=fake_executor(store_dir))
    try:
        # Verify status is preserved
        recovered = service2.status(build_id)
        assert recovered["state"] == "READY"
        assert recovered["dataset_build_id"] == build_id
        assert recovered["dataset_manifest_hash"] == manifest_hash

        # Verify artifacts are still readable and identical
        rec_root_path, rec_root_hash, _ = service2.artifact(build_id)
        assert rec_root_path.read_bytes() == root_bytes
        assert rec_root_hash == root_hash

        rec_shard_path, rec_shard_hash, _ = service2.artifact(build_id, shard_id=0)
        assert rec_shard_path.read_bytes() == shard_bytes
        assert rec_shard_hash == shard_hash

        rec_batch_path, rec_batch_hash, _ = service2.artifact(build_id, shard_id=0, batch_id=0)
        assert rec_batch_path.read_bytes() == batch_bytes
        assert rec_batch_hash == batch_hash

        # Re-submit with same idempotency key
        repeat = service2.submit(request_body, idempotency_key)
        assert repeat["dataset_build_id"] == build_id
        assert repeat["state"] == "READY"
    finally:
        service2.close()
