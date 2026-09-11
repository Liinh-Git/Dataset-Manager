from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from dataset_manager.config import DatasetBuildConfig, DatasetManagerConfig
from dataset_manager.hashing import sha256_bytes
from dataset_manager.preprocessing import Samples
from dataset_manager.schemas import DatasetBuildState
from dataset_manager.service import (
    DURABLE_RECORD_SCHEMA_VERSION,
    DatasetService,
    _BuildRecord,
    deserialize_durable_record,
    serialize_durable_record,
)
from dataset_manager.storage import HuggingFaceArtifactStore


@dataclass
class _BranchInfo:
    name: str


@dataclass
class _RepoRefs:
    branches: list[_BranchInfo]


class FakeHfApi:
    """In-memory mock of huggingface_hub.HfApi."""

    def __init__(
        self,
        writable: bool = True,
        branches: list[str] | None = None,
        fail_delete_folder: bool = False,
        fail_list_files: bool = False,
        fail_list_refs: bool = False,
    ):
        self.files: dict[str, bytes] = {}
        self.writable = writable
        self.branches = branches or ["main"]
        self.fail_delete_folder = fail_delete_folder
        self.fail_list_files = fail_list_files
        self.fail_list_refs = fail_list_refs

    def list_repo_refs(
        self, repo_id: str, repo_type: str = "dataset", token: str = ""
    ) -> _RepoRefs:
        if self.fail_list_refs:
            raise ConnectionError("Network failure listing repo refs")
        return _RepoRefs(branches=[_BranchInfo(name=b) for b in self.branches])

    def upload_folder(
        self,
        folder_path: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = "dataset",
        revision: str = "main",
        token: str = "",
        commit_message: str = "",
    ) -> None:
        if not self.writable:
            raise PermissionError("Write access denied")
        folder = Path(folder_path)
        for p in folder.rglob("*"):
            if p.is_file():
                rel = p.relative_to(folder).as_posix()
                repo_path = f"{path_in_repo}/{rel}"
                self.files[repo_path] = p.read_bytes()

    def upload_file(
        self,
        path_or_fileobj: bytes | str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = "dataset",
        revision: str = "main",
        token: str = "",
        commit_message: str = "",
    ) -> None:
        if not self.writable:
            raise PermissionError("Write access denied")
        if isinstance(path_or_fileobj, bytes):
            self.files[path_in_repo] = path_or_fileobj
        else:
            self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()

    def download_file(self, repo_id: str, filename: str, revision: str = "main") -> bytes:
        if filename not in self.files:
            raise FileNotFoundError(f"File {filename} not found in remote repo")
        return self.files[filename]

    def list_repo_files(
        self, repo_id: str, repo_type: str = "dataset", revision: str = "main", token: str = ""
    ) -> list[str]:
        if self.fail_list_files:
            raise ConnectionError("Network failure listing repo files")
        return list(self.files.keys())

    def delete_folder(
        self,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = "dataset",
        revision: str = "main",
        token: str = "",
        commit_message: str = "",
    ) -> None:
        if self.fail_delete_folder:
            raise RuntimeError("Hugging Face remote delete_folder failed")
        if not self.writable:
            raise PermissionError("Write access denied")
        prefix = f"{path_in_repo}/"
        keys_to_del = [k for k in self.files if k.startswith(prefix) or k == path_in_repo]
        for k in keys_to_del:
            del self.files[k]

    def delete_file(
        self,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = "dataset",
        revision: str = "main",
        token: str = "",
        commit_message: str = "",
    ) -> None:
        if not self.writable:
            raise PermissionError("Write access denied")
        self.files.pop(path_in_repo, None)

    def auth_check(
        self, repo_id: str, repo_type: str = "dataset", token: str = "", write: bool = True
    ) -> bool:
        if write and not self.writable:
            raise PermissionError("Write access denied")
        return True


def _sample_build_config(build_id: str = "test-build-1") -> DatasetBuildConfig:
    return DatasetBuildConfig(
        1,
        build_id,
        "cifar10",
        "CIFAR-10",
        "cifar10_quick",
        "image_classification",
        (3, 32, 32),
        "float32",
        10,
        {
            "channel_order": "NCHW",
            "scale": "uint8_to_unit",
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2470, 0.2435, 0.2616),
        },
        2,
        2,
        "seeded_permutation_round_robin",
        42,
    )


def _sample_data(count: int = 4) -> Samples:
    x = np.zeros((count, 3, 32, 32), dtype=np.float32)
    y = np.arange(count, dtype=np.int64) % 10
    sample_ids = np.arange(count, dtype=np.int64)
    return Samples(x=x, y=y, sample_ids=sample_ids)


def test_hf_store_branch_validation(tmp_path: Path):
    api = FakeHfApi(branches=["main", "staging"])
    store = HuggingFaceArtifactStore(
        local_root=tmp_path, repo_id="test/repo", token="fake", branch="main", api=api
    )
    assert store._branch == "main"

    # Missing branch must raise ValueError
    with pytest.raises(ValueError, match="does not exist in repository"):
        HuggingFaceArtifactStore(
            local_root=tmp_path, repo_id="test/repo", token="fake", branch="dev", api=api
        )

    # Remote access failure must raise RuntimeError
    failing_api = FakeHfApi(fail_list_refs=True)
    with pytest.raises(RuntimeError, match="Failed to access Hugging Face repository"):
        HuggingFaceArtifactStore(
            local_root=tmp_path, repo_id="test/repo", token="fake", branch="main", api=failing_api
        )


def test_hf_store_probe_writable(tmp_path: Path):
    writable_api = FakeHfApi(writable=True)
    store = HuggingFaceArtifactStore(
        local_root=tmp_path, repo_id="test/repo", token="fake", api=writable_api
    )
    assert store.probe_writable() is True

    readonly_api = FakeHfApi(writable=False)
    store_ro = HuggingFaceArtifactStore(
        local_root=tmp_path, repo_id="test/repo", token="fake", api=readonly_api
    )
    assert store_ro.probe_writable() is False


def test_hf_store_materialize_and_remote_verification(tmp_path: Path):
    api = FakeHfApi()
    store = HuggingFaceArtifactStore(
        local_root=tmp_path, repo_id="test/repo", token="fake", api=api
    )

    config = _sample_build_config()
    samples = _sample_data(4)
    published = store.materialize(config, samples)

    assert published.dataset_build_id == config.dataset_build_id
    assert published.manifest_path.is_file()

    expected_root = f"dataset-builds/{config.dataset_build_id}/dataset-manifest.json"
    assert expected_root in api.files
    root_data = json.loads(api.files[expected_root])
    assert root_data["dataset_build_id"] == config.dataset_build_id
    assert root_data["sample_count"] == 4


def test_hf_store_load_requires_expected_manifest_hash(tmp_path: Path):
    api = FakeHfApi()
    store = HuggingFaceArtifactStore(
        local_root=tmp_path, repo_id="test/repo", token="fake", api=api
    )
    with pytest.raises(ValueError, match="expected_manifest_hash is required"):
        store.load("any-id", expected_manifest_hash=None)


def test_hf_store_cache_miss_and_corrupt_manifest(tmp_path: Path):
    api = FakeHfApi()
    store = HuggingFaceArtifactStore(
        local_root=tmp_path, repo_id="test/repo", token="fake", api=api
    )
    config = _sample_build_config("miss-build")
    samples = _sample_data(4)
    published = store.materialize(config, samples)
    good_hash = published.dataset_manifest_hash

    fresh_local = tmp_path / "fresh_cache"
    store2 = HuggingFaceArtifactStore(
        local_root=fresh_local, repo_id="test/repo", token="fake", api=api
    )

    with pytest.raises(ValueError, match="Corrupted root manifest"):
        store2.load("miss-build", expected_manifest_hash="bad" + good_hash[3:])

    loaded = store2.load("miss-build", expected_manifest_hash=good_hash)
    assert loaded.dataset_manifest_hash == good_hash
    assert loaded.manifest_path.is_file()

    rel_batch = "shards/000/batch-000000.npz"
    local_batch = store2.resolve_artifact(loaded, rel_batch)
    assert local_batch.is_file()


def test_hf_store_corrupt_remote_shard_manifest_fails_integrity(tmp_path: Path):
    """Intentionally corrupt remote shard manifest and prove resolve_artifact fails."""
    api = FakeHfApi()
    store = HuggingFaceArtifactStore(
        local_root=tmp_path / "orig", repo_id="test/repo", token="fake", api=api
    )
    config = _sample_build_config("corrupt-shard-build")
    samples = _sample_data(4)
    published = store.materialize(config, samples)

    # Corrupt the remote shard manifest in fake HF storage
    shard_remote_key = f"dataset-builds/{config.dataset_build_id}/shards/000/shard-manifest.json"
    assert shard_remote_key in api.files
    api.files[shard_remote_key] = b'{"corrupted": true}'

    # Load on a fresh instance
    store_client = HuggingFaceArtifactStore(
        local_root=tmp_path / "client", repo_id="test/repo", token="fake", api=api
    )
    loaded = store_client.load(
        config.dataset_build_id, expected_manifest_hash=published.dataset_manifest_hash
    )

    # resolve_artifact on the corrupted shard must fail and not cache
    with pytest.raises(ValueError, match="Remote Shard Manifest SHA-256 mismatch"):
        store_client.resolve_artifact(loaded, "shards/000/shard-manifest.json")

    cached_file = store_client._local_store._safe(
        loaded.directory, "shards/000/shard-manifest.json"
    )
    assert not cached_file.exists()


def test_hf_store_corrupt_remote_batch_payload_fails_integrity(tmp_path: Path):
    """Intentionally corrupt remote batch payload and prove resolve_artifact fails."""
    api = FakeHfApi()
    store = HuggingFaceArtifactStore(
        local_root=tmp_path / "orig", repo_id="test/repo", token="fake", api=api
    )
    config = _sample_build_config("corrupt-batch-build")
    samples = _sample_data(4)
    published = store.materialize(config, samples)

    # Corrupt the remote batch payload in fake HF storage
    batch_remote_key = f"dataset-builds/{config.dataset_build_id}/shards/000/batch-000000.npz"
    assert batch_remote_key in api.files
    orig_payload = api.files[batch_remote_key]
    api.files[batch_remote_key] = orig_payload + b"corruption"

    # Load on a fresh instance
    store_client = HuggingFaceArtifactStore(
        local_root=tmp_path / "client", repo_id="test/repo", token="fake", api=api
    )
    loaded = store_client.load(
        config.dataset_build_id, expected_manifest_hash=published.dataset_manifest_hash
    )

    with pytest.raises(ValueError, match=r"Remote Batch (byte size|SHA-256) mismatch"):
        store_client.resolve_artifact(loaded, "shards/000/batch-000000.npz")

    cached_file = store_client._local_store._safe(loaded.directory, "shards/000/batch-000000.npz")
    assert not cached_file.exists()


def test_hf_store_purge_fails_closed_on_remote_error(tmp_path: Path):
    """Verify remote deletion failure causes purge to fail closed and keeps build in DELETING."""
    api = FakeHfApi(fail_delete_folder=True)
    config = DatasetManagerConfig(
        storage_backend="huggingface",
        hf_repo_id="test/repo",
        hf_token="fake-token",
        store_dir=str(tmp_path / "store"),
        temp_dir=str(tmp_path / "temp"),
    )

    from dataset_manager import storage

    orig_init = storage.HuggingFaceArtifactStore.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["api"] = api
        orig_init(self, *args, **kwargs)

    storage.HuggingFaceArtifactStore.__init__ = patched_init
    try:
        service = DatasetService(config, start_worker=False)
        # Pre-populate remote artifact file so remote folder exists
        build_id = "purge-fail-build"
        api.files[f"dataset-builds/{build_id}/dataset-manifest.json"] = b"{}"

        # Create a build record directly in DEPRECATED state
        record = _BuildRecord(
            dataset_build_id=build_id,
            request={"shard_count": 2, "batch_size": 2, "profile": "cifar10_quick"},
            request_fingerprint="fp-1",
            idempotency_key_hash=sha256_bytes(b"key-1"),
            state=DatasetBuildState.DEPRECATED,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            dataset_manifest_hash="hash-123",
        )
        service._records[build_id] = record
        service._persist(record)

        # purge must raise RuntimeError because delete_folder fails
        with pytest.raises(RuntimeError, match="Hugging Face remote delete_folder failed"):
            service.purge(build_id, command_id="cmd-purge-fail")

        # Build must NOT be DELETED; it must remain in DELETING
        assert service._records[build_id].state == DatasetBuildState.DELETING
        service.close()
    finally:
        storage.HuggingFaceArtifactStore.__init__ = orig_init


def test_hf_store_load_records_fails_on_remote_outage(tmp_path: Path):
    """In huggingface mode, load_records must raise on remote failure, not silently fall back."""
    api = FakeHfApi(fail_list_files=True)
    store = HuggingFaceArtifactStore(
        local_root=tmp_path, repo_id="test/repo", token="fake", api=api
    )
    with pytest.raises(RuntimeError, match="Failed to list remote records"):
        store.load_records()


def test_hf_store_load_records_prunes_stale_local_cache(tmp_path: Path):
    """Remote listing must prune local metadata records that no longer exist on remote."""
    api = FakeHfApi()
    store = HuggingFaceArtifactStore(
        local_root=tmp_path, repo_id="test/repo", token="fake", api=api
    )

    # Pre-populate a stale local record that does NOT exist on remote HF
    stale_file = store._local_store._metadata / "stale-build.json"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text('{"dataset_build_id": "stale-build"}', encoding="utf-8")
    assert stale_file.exists()

    # Now add a valid record to remote
    valid_record = {
        "dataset_build_id": "valid-remote-build",
        "state": "READY",
        "idempotency_key_hash": "hash-valid",
    }
    api.files[".service/builds/valid-remote-build.json"] = json.dumps(valid_record).encode("utf-8")

    records = store.load_records()
    record_ids = [r["dataset_build_id"] for r in records]

    # Valid remote record must be present; stale local record must be pruned and not resurrected
    assert "valid-remote-build" in record_ids
    assert "stale-build" not in record_ids
    assert not stale_file.exists()


def test_durable_record_serialization_whitelist():
    record = _BuildRecord(
        dataset_build_id="build-whitelisted",
        request={"cifar": True},
        request_fingerprint="fp123",
        idempotency_key_hash="hash456",
        state="COMPLETED",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        idempotency_key="secret-raw-key",
        raw_workspace="D:\\local\\temp\\workspace",
        error={"code": "ERR", "message": "Failed at D:\\local\\temp\\path on drive C:\\data"},
    )

    local_data = serialize_durable_record(record, is_remote=False)
    assert local_data["raw_workspace"] == "D:\\local\\temp\\workspace"
    assert local_data["idempotency_key"] == "secret-raw-key"

    remote_data = serialize_durable_record(record, is_remote=True)
    assert "raw_workspace" not in remote_data
    assert "idempotency_key" not in remote_data
    assert remote_data["idempotency_key_hash"] == "hash456"

    msg = remote_data["error"]["message"]
    assert "D:\\" not in msg
    assert "C:\\" not in msg
    assert "<path>" in msg


def test_durable_record_deserialization_migration():
    legacy_data = {
        "dataset_build_id": "legacy-build",
        "request": {},
        "request_fingerprint": "legacy-fp",
        "idempotency_key": "raw-key-value",
        "state": "READY",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    rec = deserialize_durable_record(legacy_data)
    assert rec.schema_version == DURABLE_RECORD_SCHEMA_VERSION
    assert rec.idempotency_key_hash == sha256_bytes(b"raw-key-value")


def test_durable_record_deserialization_strict_rejections():
    """Verify rejection of records without idempotency hash/key, or with future schema versions."""
    # 1. No idempotency_key_hash and no idempotency_key
    invalid_data = {
        "dataset_build_id": "invalid-build",
        "request": {},
        "request_fingerprint": "fp-only",
        "state": "READY",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    with pytest.raises(
        ValueError, match="has neither idempotency_key_hash nor legacy idempotency_key"
    ):
        deserialize_durable_record(invalid_data)

    # 2. Future schema version
    future_data = {
        "schema_version": 99,
        "dataset_build_id": "future-build",
        "request": {},
        "request_fingerprint": "fp-future",
        "idempotency_key_hash": "hash-future",
        "state": "READY",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    with pytest.raises(ValueError, match="Unsupported durable record schema version: 99"):
        deserialize_durable_record(future_data)


def test_hf_store_purge_crash_recovery_when_remote_folder_already_absent(tmp_path: Path):
    """Crash recovery: DELETING metadata exists, remote artifact folder is already absent.

    Restarting DatasetService must recover cleanly to DELETED, and persist DELETED to HF.
    """
    api = FakeHfApi()
    build_id = "crash-recover-build"
    manifest_hash = "abc123hash"

    # Pre-populate durable remote metadata in DELETING state
    record_data = {
        "schema_version": DURABLE_RECORD_SCHEMA_VERSION,
        "dataset_build_id": build_id,
        "request": {"shard_count": 2, "batch_size": 2, "profile": "cifar10_quick"},
        "request_fingerprint": "fp-crash",
        "idempotency_key_hash": sha256_bytes(b"key-crash"),
        "state": DatasetBuildState.DELETING.value,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "dataset_manifest_hash": manifest_hash,
    }
    api.files[f".service/builds/{build_id}.json"] = json.dumps(record_data).encode("utf-8")

    # Remote artifact folder dataset-builds/{build_id} is already absent (not in api.files)
    assert not any(k.startswith(f"dataset-builds/{build_id}") for k in api.files)

    config = DatasetManagerConfig(
        storage_backend="huggingface",
        hf_repo_id="test/repo",
        hf_token="fake-token",
        store_dir=str(tmp_path / "store"),
        temp_dir=str(tmp_path / "temp"),
    )

    from dataset_manager import storage

    orig_init = storage.HuggingFaceArtifactStore.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["api"] = api
        orig_init(self, *args, **kwargs)

    storage.HuggingFaceArtifactStore.__init__ = patched_init
    try:
        # Start DatasetService; _load() executes crash recovery
        service = DatasetService(config, start_worker=False)

        # 1. Recovery completes to DELETED
        assert service.status(build_id)["state"] == "DELETED"

        # 2. Durable remote metadata becomes DELETED
        remote_meta = json.loads(api.files[f".service/builds/{build_id}.json"])
        assert remote_meta["state"] == "DELETED"

        service.close()
    finally:
        storage.HuggingFaceArtifactStore.__init__ = orig_init


def test_hf_store_resolve_artifact_corrupt_cache_hit_self_heals(tmp_path: Path):
    """Corrupted local cache hit is detected, unlinked, and refetched from HF."""
    api = FakeHfApi()
    store = HuggingFaceArtifactStore(
        local_root=tmp_path / "store", repo_id="test/repo", token="fake", api=api
    )
    config = _sample_build_config("cache-heal-build")
    samples = _sample_data(4)
    published = store.materialize(config, samples)

    rel_batch = "shards/000/batch-000000.npz"
    local_batch = store.resolve_artifact(published, rel_batch)
    assert local_batch.is_file()
    valid_bytes = local_batch.read_bytes()

    # Intentionally corrupt the local cache file
    local_batch.write_bytes(b"corrupted local content")
    assert local_batch.read_bytes() != valid_bytes

    # resolve_artifact on the corrupt cache hit must detect corruption, refetch and heal
    healed_path = store.resolve_artifact(published, rel_batch)
    assert healed_path.is_file()
    assert healed_path.read_bytes() == valid_bytes
