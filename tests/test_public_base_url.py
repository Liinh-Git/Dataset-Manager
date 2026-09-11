"""Verify PUBLIC_BASE_URL multi-machine URL generation contract."""

from pathlib import Path

import numpy as np

from dataset_manager.config import DatasetBuildConfig, DatasetManagerConfig
from dataset_manager.preprocessing import Preprocessor
from dataset_manager.service import BuildExecutionResult, DatasetService
from dataset_manager.storage import DatasetStorage


def fake_executor(tmp_path: Path):
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
        published = DatasetStorage(tmp_path / "store").materialize(config, samples)
        return BuildExecutionResult(published)

    return _exec


def test_public_base_url_in_returned_uris(tmp_path: Path):
    external_url = "https://external-dataset-manager.prod"
    config = DatasetManagerConfig(
        host="0.0.0.0",
        port=9200,
        store_dir=str(tmp_path / "store"),
        temp_dir=str(tmp_path / "temp"),
        public_base_url=external_url,
    )
    service = DatasetService(config, executor=fake_executor(tmp_path))
    try:
        request_body = {
            "command_id": "cmd-public-test",
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
        sub = service.submit(request_body, "key-public-url")
        build_id = sub["dataset_build_id"]

        import time

        start = time.time()
        while time.time() - start < 10:
            st = service.status(build_id)
            if st["state"] in ("REGISTERING", "READY"):
                break
            time.sleep(0.05)

        ack = service.acknowledge_registration(
            build_id,
            st["dataset_manifest_hash"],
            "reg-public-url",
            "2026-09-11T12:00:00Z",
        )
        assert ack["state"] == "READY"

        artifact_base_url = ack["artifact_base_url"]
        manifest_uri = ack["manifest_uri"]

        assert artifact_base_url.startswith(external_url)
        assert manifest_uri.startswith(external_url)
        assert "127.0.0.1" not in artifact_base_url
        assert "localhost" not in artifact_base_url
        assert "127.0.0.1" not in manifest_uri
        assert "localhost" not in manifest_uri
    finally:
        service.close()
