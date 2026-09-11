"""Dataset Manager lifecycle, queue, HTTP contract, restart, and artifact tests."""

import hashlib
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dataset_manager.app import create_app
from dataset_manager.config import DatasetBuildConfig, DatasetManagerConfig
from dataset_manager.preprocessing import Preprocessor
from dataset_manager.schemas import DatasetBuildState
from dataset_manager.service import (
    BuildExecutionResult,
    DatasetBuildPipeline,
    DatasetService,
    DatasetServiceError,
)
from dataset_manager.storage import DatasetStorage


def build_request(command_id: str = "command-1", seed: int = 42) -> dict:
    return {
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
        "partition_seed": seed,
        "command_id": command_id,
    }


def manager_config(tmp_path, queue_capacity: int = 2) -> DatasetManagerConfig:
    return DatasetManagerConfig(
        store_dir=str(tmp_path / "store"),
        temp_dir=str(tmp_path / "source"),
        queue_capacity=queue_capacity,
        public_base_url="http://dataset-manager:9200",
    )


class TestExecutor:
    __test__ = False

    def __init__(self, config: DatasetManagerConfig, block: threading.Event | None = None):
        self.storage = DatasetStorage(config.store_dir)
        self.temp_root = config.temp_dir
        self.block = block
        self.entered = threading.Event()
        self.order: list[str] = []

    def __call__(self, build_id, request, update):
        self.order.append(build_id)
        update(DatasetBuildState.IMPORTING, "IMPORTING", 0.1)
        self.entered.set()
        if self.block is not None:
            assert self.block.wait(5)
        raw = np.arange(9 * 3 * 32 * 32, dtype=np.uint32).astype(np.uint8)
        raw = raw.reshape(9, 3, 32, 32)
        labels = np.arange(9, dtype=np.int64)
        normalization = request["normalization"]
        samples = Preprocessor(
            (3, 32, 32),
            10,
            tuple(normalization["mean"]),
            tuple(normalization["std"]),
        ).transform(raw, labels)
        update(DatasetBuildState.PREPROCESSING, "PREPROCESSING", 0.4)
        config = DatasetBuildConfig(
            1,
            build_id,
            "cifar10",
            "CIFAR-10",
            request["profile"],
            "image_classification",
            (3, 32, 32),
            "float32",
            10,
            {
                "channel_order": "NCHW",
                "scale": "uint8_to_unit",
                "mean": normalization["mean"],
                "std": normalization["std"],
            },
            request["batch_size"],
            request["shard_count"],
            "seeded_permutation_round_robin",
            request["partition_seed"],
        )
        update(DatasetBuildState.MATERIALIZING, "MATERIALIZING", 0.7)
        published = self.storage.materialize(config, samples)
        update(DatasetBuildState.VERIFYING, "VERIFYING", 0.9)
        raw_workspace = Path(self.temp_root) / build_id
        raw_workspace.mkdir(parents=True)
        (raw_workspace / "source.bin").write_bytes(b"retained until READY")
        return BuildExecutionResult(published, raw_workspace)


class FailingExecutor:
    def __init__(self, config: DatasetManagerConfig):
        self.temp_root = Path(config.temp_dir)

    def __call__(self, build_id, _request, update):
        update(DatasetBuildState.IMPORTING, "IMPORTING", 0.1)
        workspace = self.temp_root / build_id
        workspace.mkdir(parents=True)
        (workspace / "partial.bin").write_bytes(b"partial")
        raise OSError("source interrupted")


def wait_for_state(service: DatasetService, build_id: str, expected: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = service.status(build_id)
        if value["state"] == expected:
            return value
        time.sleep(0.01)
    pytest.fail(f"build {build_id} did not reach {expected}")


def test_source_cache_requires_official_checksum(tmp_path, monkeypatch):
    cached = tmp_path / "cached.tar.gz"
    cached.write_bytes(b"wrong")
    destination = tmp_path / "download.part"
    downloaded = b"fresh"
    config = DatasetManagerConfig(
        store_dir=str(tmp_path / "store"),
        temp_dir=str(tmp_path / "source"),
        download_parallelism=1,
    )
    pipeline = DatasetBuildPipeline(config, DatasetStorage(config.store_dir))
    monkeypatch.setenv("PBL4_CIFAR10_SOURCE_ARCHIVE", str(cached))
    monkeypatch.setattr(
        "dataset_manager.service._CIFAR10_BINARY_ARCHIVE_BYTES", len(cached.read_bytes())
    )
    monkeypatch.setattr(
        "dataset_manager.service._CIFAR10_BINARY_ARCHIVE_MD5",
        hashlib.md5(b"official", usedforsecurity=False).hexdigest(),
    )

    class HeadResponse:
        def __init__(self):
            self.headers = {"Content-Length": str(len(downloaded)), "Accept-Ranges": "none"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return "https://example.invalid/cifar.tar.gz"

    monkeypatch.setattr("dataset_manager.service._CIFAR10_BINARY_ARCHIVE_BYTES", len(downloaded))
    monkeypatch.setattr(pipeline, "_download_sequential", lambda path: path.write_bytes(downloaded))
    with patch("urllib.request.urlopen", return_value=HeadResponse()):
        pipeline._download(destination)
    assert destination.read_bytes() == downloaded


def test_single_active_fifo_queue_saturation_and_no_orphan(tmp_path):
    release = threading.Event()
    config = manager_config(tmp_path, queue_capacity=1)
    executor = TestExecutor(config, release)
    service = DatasetService(config, executor)
    first = service.submit(build_request("one"), "key-one")
    assert executor.entered.wait(5)
    second = service.submit(build_request("two"), "key-two")
    assert second["state"] == DatasetBuildState.QUEUED
    with pytest.raises(DatasetServiceError) as rejected:
        service.submit(build_request("three"), "key-three")
    assert rejected.value.code == "BUILD_QUEUE_FULL"
    assert rejected.value.retryable
    assert len(list((tmp_path / "store" / ".service" / "builds").glob("*.json"))) == 2
    release.set()
    wait_for_state(service, first["dataset_build_id"], DatasetBuildState.REGISTERING)
    wait_for_state(service, second["dataset_build_id"], DatasetBuildState.REGISTERING)
    assert executor.order == [first["dataset_build_id"], second["dataset_build_id"]]
    service.close()


def test_create_idempotency_and_registering_survive_restart(tmp_path):
    config = manager_config(tmp_path)
    executor = TestExecutor(config)
    service = DatasetService(config, executor)
    first = service.submit(build_request(), "same-key")
    replay = service.submit(build_request(), "same-key")
    assert replay["dataset_build_id"] == first["dataset_build_id"]
    with pytest.raises(DatasetServiceError, match="different request"):
        service.submit(build_request(seed=99), "same-key")
    registering = wait_for_state(service, first["dataset_build_id"], DatasetBuildState.REGISTERING)
    assert registering["dataset_manifest_hash"]
    service.close()

    restarted = DatasetService(config, executor, start_worker=False)
    assert restarted.status(first["dataset_build_id"])["state"] == DatasetBuildState.REGISTERING
    assert (
        restarted.submit(build_request(), "same-key")["dataset_build_id"]
        == first["dataset_build_id"]
    )
    restarted.close()


def test_rebuild_revalidates_profile_and_failed_purge_removes_raw_workspace(tmp_path):
    config = manager_config(tmp_path)
    service = DatasetService(config, TestExecutor(config))
    created = service.submit(build_request(), "original")
    wait_for_state(service, created["dataset_build_id"], DatasetBuildState.REGISTERING)
    with pytest.raises(DatasetServiceError) as invalid:
        service.rebuild(
            created["dataset_build_id"],
            {"command_id": "invalid-rebuild", "input_shape": [1, 32, 32]},
            "invalid-rebuild-key",
        )
    assert invalid.value.code == "INVALID_REQUEST"
    assert invalid.value.status_code == 400
    service.close()

    failing = DatasetService(config, FailingExecutor(config))
    failed = failing.submit(build_request("failed"), "failed-key")
    wait_for_state(failing, failed["dataset_build_id"], DatasetBuildState.FAILED)
    workspace = Path(config.temp_dir) / failed["dataset_build_id"]
    assert workspace.is_dir()
    purged = failing.purge(failed["dataset_build_id"], "purge-failed")
    assert purged["state"] == DatasetBuildState.DELETED
    assert not workspace.exists()
    failing.close()


def test_http_registration_visibility_lifecycle_and_purge(tmp_path):
    config = manager_config(tmp_path)
    executor = TestExecutor(config)
    service = DatasetService(config, executor)
    app = create_app(config, service)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/dataset-builds",
            headers={"Idempotency-Key": "create-key", "X-Request-Id": "request-1"},
            json=build_request(),
        )
        assert created.status_code == 202
        assert created.json()["request_id"] == "request-1"
        build_id = created.json()["data"]["dataset_build_id"]
        registering = wait_for_state(service, build_id, DatasetBuildState.REGISTERING)
        digest = registering["dataset_manifest_hash"]
        artifact_base = f"http://dataset-manager:9200/artifacts/v1/dataset-builds/{build_id}"
        assert registering["artifact_base_url"] == artifact_base
        assert registering["manifest_uri"] == f"{artifact_base}/manifest.json"

        root = client.get(f"/artifacts/v1/dataset-builds/{build_id}/manifest.json")
        assert root.status_code == 200
        assert root.headers["content-length"] == str(len(root.content))
        assert root.headers["etag"] == f'"{digest}"'
        denied = client.get(f"/artifacts/v1/dataset-builds/{build_id}/shards/0/manifest.json")
        assert denied.status_code == 409
        assert denied.json()["error"]["code"] == "BUILD_NOT_READY"
        assert (Path(config.temp_dir) / build_id / "source.bin").is_file()

        wrong = client.post(
            f"/api/v1/dataset-builds/{build_id}/registration-ack",
            json={
                "dataset_manifest_hash": "0" * 64,
                "registration_id": "registration-1",
                "catalog_persisted_at": "2026-09-08T00:00:00Z",
            },
        )
        assert wrong.status_code == 409
        assert service.status(build_id)["state"] == DatasetBuildState.REGISTERING
        ack = {
            "dataset_manifest_hash": digest,
            "registration_id": "registration-1",
            "catalog_persisted_at": "2026-09-08T00:00:00Z",
        }
        ready = client.post(f"/api/v1/dataset-builds/{build_id}/registration-ack", json=ack)
        assert ready.status_code == 200
        assert ready.json()["data"]["state"] == DatasetBuildState.READY
        assert not (Path(config.temp_dir) / build_id).exists()
        assert (
            client.post(f"/api/v1/dataset-builds/{build_id}/registration-ack", json=ack).status_code
            == 200
        )
        conflict = {**ack, "registration_id": "registration-2"}
        assert (
            client.post(
                f"/api/v1/dataset-builds/{build_id}/registration-ack", json=conflict
            ).status_code
            == 409
        )

        shard = client.get(f"/artifacts/v1/dataset-builds/{build_id}/shards/0/manifest.json")
        batch = client.get(f"/artifacts/v1/dataset-builds/{build_id}/shards/0/batches/0")
        assert shard.status_code == batch.status_code == 200
        assert batch.headers["content-type"] == "application/octet-stream"
        assert (
            client.get(
                f"/artifacts/v1/dataset-builds/{build_id}/shards/99/manifest.json"
            ).status_code
            == 404
        )

        deprecate = client.post(
            f"/api/v1/dataset-builds/{build_id}/deprecate", json={"reason": "old"}
        )
        assert deprecate.json()["data"]["state"] == DatasetBuildState.DEPRECATED
        assert (
            client.post(f"/api/v1/dataset-builds/{build_id}/deprecate", json={}).status_code == 200
        )
        purge = client.post(
            f"/api/v1/dataset-builds/{build_id}/purge",
            json={"command_id": "purge-1", "force": False},
        )
        assert purge.json()["data"]["state"] == DatasetBuildState.DELETED
        assert (
            client.post(
                f"/api/v1/dataset-builds/{build_id}/purge",
                json={"command_id": "purge-1", "force": False},
            ).status_code
            == 200
        )
        assert (
            client.get(f"/artifacts/v1/dataset-builds/{build_id}/manifest.json").status_code == 409
        )
    service.close()


def test_invalid_api_request_has_canonical_error_envelope(tmp_path):
    config = manager_config(tmp_path)
    service = DatasetService(config, TestExecutor(config), start_worker=False)
    with TestClient(create_app(config, service)) as client:
        response = client.post(
            "/api/v1/dataset-builds",
            headers={"Idempotency-Key": "bad"},
            json={**build_request(), "shard_count": 2},
        )
    assert response.status_code == 400
    assert response.json()["data"] is None
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    service.close()


# ── Issue C: /healthz canonical local-service tests ───────────────────────────


def test_c1_healthz_returns_all_required_fields(tmp_path):
    """C1: healthy empty service returns all canonical fields."""
    config = manager_config(tmp_path)
    service = DatasetService(config, TestExecutor(config), start_worker=False)
    with TestClient(create_app(config, service)) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "ok"
    assert data["service"] == "dataset-manager"
    assert "version" in data
    assert data["queue_depth"] == 0
    assert data["active_build_id"] is None
    assert data["storage_writable"] is True
    service.close()


def test_c2_healthz_reflects_queued_and_active_build(tmp_path):
    """C2: queue depth and active_build_id reflected in health snapshot."""
    release = threading.Event()
    config = manager_config(tmp_path, queue_capacity=2)
    executor = TestExecutor(config, release)
    service = DatasetService(config, executor)
    first = service.submit(build_request("one"), "key-one")
    assert executor.entered.wait(5)
    # One build is active, none in queue yet
    h = service.health()
    assert h["active_build_id"] == first["dataset_build_id"]
    assert h["queue_depth"] == 0
    service.submit(build_request("two"), "key-two")
    h2 = service.health()
    assert h2["queue_depth"] == 1
    release.set()
    service.close()


def test_c3_storage_not_writable_reports_degraded(tmp_path, monkeypatch):
    """C3: if storage probe fails, status=degraded and storage_writable=False."""
    config = manager_config(tmp_path)
    service = DatasetService(config, TestExecutor(config), start_worker=False)

    def fail_probe():
        raise OSError("read-only")

    monkeypatch.setattr(service, "_probe_storage_writable", lambda: False)
    h = service.health()
    assert h["status"] == "degraded"
    assert h["storage_writable"] is False
    service.close()


def test_c4_registering_does_not_make_health_degraded(tmp_path):
    """C4: a build in REGISTERING state does NOT degrade health (no Backend dependency)."""
    config = manager_config(tmp_path)
    executor = TestExecutor(config)
    service = DatasetService(config, executor)
    created = service.submit(build_request(), "reg-key")
    wait_for_state(service, created["dataset_build_id"], DatasetBuildState.REGISTERING)
    # Service is still locally healthy even though Backend ACK hasn't arrived
    h = service.health()
    assert h["status"] == "ok"
    assert h["storage_writable"] is True
    service.close()
