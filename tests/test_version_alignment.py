"""Version-alignment coverage for DBS boundaries, WAN hardening and memory use."""

import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dataset_manager.app import create_app
from dataset_manager.config import DatasetBuildConfig, DatasetManagerConfig
from dataset_manager.importer import DatasetImporter
from dataset_manager.preprocessing import PreprocessedSampleSource, Preprocessor
from dataset_manager.schemas import DatasetBuildState
from dataset_manager.service import BuildExecutionResult, DatasetService
from dataset_manager.storage import DatasetStorage


def _request() -> dict[str, object]:
    return {
        "command_id": "alignment-create",
        "source": {
            "type": "cifar10_download",
            "dataset_name": "cifar10",
            "version": "binary-v1",
        },
        "profile": "CNN_IMAGE_CLASSIFICATION_V1",
        "input_shape": [3, 32, 32],
        "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "batch_size": 2,
        "shard_count": 3,
        "partition_seed": 42,
    }


def _config(build_id: str) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        1,
        build_id,
        "cifar10",
        "CIFAR-10",
        "CNN_IMAGE_CLASSIFICATION_V1",
        "image_classification",
        (3, 32, 32),
        "float32",
        10,
        {
            "channel_order": "NCHW",
            "scale": "uint8_to_unit",
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
        },
        2,
        3,
        "seeded_permutation_round_robin",
        42,
    )


def _cifar_file(path: Path, count: int = 9) -> Path:
    labels = (np.arange(count, dtype=np.uint8) % 10)[:, None]
    pixels = np.arange(count * 3072, dtype=np.uint32).astype(np.uint8).reshape(count, 3072)
    path.write_bytes(np.concatenate((labels, pixels), axis=1).tobytes())
    return path


def test_lazy_preprocessing_is_byte_identical_to_eager_path(tmp_path: Path):
    source_file = _cifar_file(tmp_path / "data_batch.bin")
    importer = DatasetImporter()
    preprocessor = Preprocessor((3, 32, 32), 10, (0.5,) * 3, (0.5,) * 3)

    eager_raw = importer.import_cifar10_binary((source_file,))
    eager = preprocessor.transform(eager_raw.x, eager_raw.y, sample_ids=eager_raw.sample_ids)
    lazy = PreprocessedSampleSource(importer.open_cifar10_binary((source_file,)), preprocessor)
    selected = np.array([8, 0, 4, 2], dtype=np.int64)
    lazy_selected = lazy.take(selected)
    np.testing.assert_array_equal(lazy_selected.sample_ids, eager.sample_ids[selected])
    np.testing.assert_array_equal(lazy_selected.y, eager.y[selected])
    np.testing.assert_allclose(lazy_selected.x, eager.x[selected], rtol=0, atol=0)

    first = DatasetStorage(tmp_path / "eager").materialize(_config("same-build"), eager)
    second = DatasetStorage(tmp_path / "lazy").materialize(_config("same-build"), lazy)
    assert first.dataset_manifest_hash == second.dataset_manifest_hash
    eager_files = {
        path.relative_to(first.directory).as_posix(): path.read_bytes()
        for path in first.directory.rglob("*")
        if path.is_file()
    }
    lazy_files = {
        path.relative_to(second.directory).as_posix(): path.read_bytes()
        for path in second.directory.rglob("*")
        if path.is_file()
    }
    assert lazy_files == eager_files


def test_dataset_service_has_no_hugging_face_url_layout_knowledge():
    source = inspect.getsource(DatasetService)
    assert "huggingface.co" not in source
    assert "/datasets/" not in source
    assert "/resolve/" not in source


def test_concurrent_all_shards_download_and_offline_cache(tmp_path: Path):
    config = DatasetManagerConfig(
        store_dir=str(tmp_path / "store"),
        temp_dir=str(tmp_path / "temp"),
        public_base_url="http://dataset-manager:9200",
    )

    def executor(build_id, request, update):
        raw = np.arange(15 * 3 * 32 * 32, dtype=np.uint32).astype(np.uint8)
        raw = raw.reshape(15, 3, 32, 32)
        prepared = Preprocessor((3, 32, 32), 10, (0.5,) * 3, (0.5,) * 3).transform(
            raw, np.arange(15, dtype=np.int64) % 10
        )
        update(DatasetBuildState.MATERIALIZING, "MATERIALIZING", 0.6)
        published = DatasetStorage(config.store_dir).materialize(_config(build_id), prepared)
        update(DatasetBuildState.VERIFYING, "VERIFYING", 0.9)
        return BuildExecutionResult(published)

    service = DatasetService(config, executor=executor)
    build_id = service.submit(_request(), "all-shards")["dataset_build_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = service.status(build_id)
        if status["state"] == DatasetBuildState.REGISTERING:
            break
        time.sleep(0.01)
    else:
        pytest.fail("build did not reach REGISTERING")
    service.acknowledge_registration(
        build_id,
        status["dataset_manifest_hash"],
        "registration-all-shards",
        "2026-09-24T00:00:00Z",
    )

    app = create_app(config, service)
    worker_caches = [tmp_path / f"worker-{index}" for index in range(5)]

    def cache_all(destination: Path) -> dict[str, bytes]:
        with TestClient(app) as client:
            base = f"/artifacts/v1/dataset-builds/{build_id}"
            root_response = client.get(f"{base}/manifest.json")
            assert root_response.status_code == 200
            assert root_response.headers["content-length"] == str(len(root_response.content))
            root = root_response.json()
            assert "worker_id" not in json.dumps(root)
            cached = {"manifest.json": root_response.content}
            partial_batch_seen = False
            for shard_ref in root["shards"]:
                shard_id = shard_ref["shard_id"]
                shard_response = client.get(f"{base}/shards/{shard_id}/manifest.json")
                assert shard_response.status_code == 200
                shard = shard_response.json()
                assert "worker_id" not in json.dumps(shard)
                for batch in shard["batches"]:
                    partial_batch_seen |= batch["sample_count"] < root["batch_size"]
                    batch_id = batch["batch_id"]
                    response = client.get(f"{base}/shards/{shard_id}/batches/{batch_id}")
                    assert response.status_code == 200
                    assert response.headers["content-length"] == str(len(response.content))
                    cached[f"{shard_id}/{batch_id}.npz"] = response.content
            assert partial_batch_seen
            destination.mkdir()
            for name, content in cached.items():
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            return cached

    with ThreadPoolExecutor(max_workers=len(worker_caches)) as pool:
        results = list(pool.map(cache_all, worker_caches))
    assert all(result == results[0] for result in results[1:])

    service.close()
    for cache in worker_caches:
        batch_path = next(cache.rglob("*.npz"))
        with np.load(batch_path, allow_pickle=False) as batch:
            assert set(batch.files) == {"x", "y", "sample_ids"}
            assert len(batch["x"]) > 0
