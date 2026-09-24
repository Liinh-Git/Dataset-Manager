"""Verified Dataset Build materialization and atomic immutable publication.

Supports Local filesystem and Hugging Face Dataset repository durable backends.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote
from uuid import uuid4

import numpy as np

from dataset_manager.batch_builder import BatchBuilder
from dataset_manager.config import DatasetBuildConfig
from dataset_manager.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from dataset_manager.manifest import DatasetManifest, ManifestBuilder
from dataset_manager.partitioner import Partitioner
from dataset_manager.preprocessing import SampleSource
from dataset_manager.profiles import validate_materialized_batch_counts

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PublishedDatasetBuild:
    dataset_build_id: str
    dataset_manifest_hash: str
    directory: Path
    manifest_path: Path
    artifact_base_url: str | None = None
    root_manifest_path: str | None = None

    def __post_init__(self) -> None:
        if (self.artifact_base_url is None) != (self.root_manifest_path is None):
            raise ValueError(
                "Artifact base URL and root manifest path must be provided together"
            )

    @property
    def lifecycle_state(self) -> str:
        # Catalog acknowledgement is deliberately outside this artifact phase.
        return "REGISTERING"


class ArtifactStore(Protocol):
    """Abstract durable storage contract for dataset artifacts and service records."""

    def materialize(
        self, config: DatasetBuildConfig, samples: SampleSource
    ) -> PublishedDatasetBuild: ...

    def load(
        self, dataset_build_id: str, expected_manifest_hash: str | None = None
    ) -> PublishedDatasetBuild: ...

    def verify(self, published: PublishedDatasetBuild) -> DatasetManifest: ...

    def resolve_artifact(self, published: PublishedDatasetBuild, relative: str) -> Path: ...

    def purge(self, dataset_build_id: str, dataset_manifest_hash: str) -> None: ...

    def persist_record(self, record_data: dict[str, Any]) -> None: ...

    def load_records(self) -> list[dict[str, Any]]: ...

    def probe_writable(self) -> bool: ...


class LocalArtifactStore:
    """Local V1 store. Temporary and final directories share the same parent volume."""

    def __init__(self, root: Path):
        self._root = Path(root).resolve()
        self._temporary_root = self._root / ".tmp"
        self._metadata = self._root / ".service" / "builds"
        self._metadata.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _directory_key(dataset_build_id: str) -> str:
        if not dataset_build_id:
            raise ValueError("Empty Dataset Build identity")
        return sha256_bytes(dataset_build_id.encode("utf-8"))

    @staticmethod
    def _safe(root: Path, relative: str) -> Path:
        # Explicitly reject invalid path syntax before any OS resolution.
        if not isinstance(relative, str) or not relative:
            raise ValueError("Artifact relative path must be a nonempty string")
        # Reject NUL bytes (null-byte injection)
        if "\x00" in relative:
            raise ValueError("Artifact path contains NUL byte")
        # Reject backslashes (cross-platform traversal ambiguity)
        if "\\" in relative:
            raise ValueError("Artifact path must use forward-slash separators")
        # Reject absolute paths (leading /, Windows drive letters like C:)
        if relative.startswith("/") or (len(relative) >= 2 and relative[1] == ":"):
            raise ValueError("Artifact path must be relative, not absolute")
        # Reject '..' segments to prevent parent-directory traversal
        parts = relative.split("/")
        if any(part == ".." for part in parts):
            raise ValueError("Artifact path must not contain '..' segments")
        if any(part == "" for part in parts):
            raise ValueError("Artifact path must not contain empty segments")
        candidate = (root / Path(*parts)).resolve()
        if candidate == root or root not in candidate.parents:
            raise ValueError("Artifact path escapes build root")
        current = root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Artifact path crosses a symlink")
        return candidate

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            if stream.write(content) != len(content):
                raise OSError("Short artifact write")
            stream.flush()
            os.fsync(stream.fileno())

    def materialize(
        self, config: DatasetBuildConfig, samples: SampleSource
    ) -> PublishedDatasetBuild:
        if samples.sample_count <= 0 or tuple(samples.input_shape) != config.input_shape:
            raise ValueError("Invalid canonical sample collection")
        self._root.mkdir(parents=True, exist_ok=True)
        self._temporary_root.mkdir(exist_ok=True)
        final = self._root / self._directory_key(config.dataset_build_id)
        workspace = self._temporary_root / uuid4().hex
        workspace.mkdir()
        try:
            partitions = Partitioner().partition(
                samples.sample_count, config.shard_count, config.partition_seed
            )
            if any(not len(partition) for partition in partitions):
                raise ValueError("Dataset Build configuration would create an empty shard")
            physical = BatchBuilder().split(partitions, config.batch_size)
            batch_counts = tuple(len(shard) for shard in physical)
            validate_materialized_batch_counts(config.profile, batch_counts)
            manifest_builder = ManifestBuilder()
            shard_entries = []
            for shard_id, batches in enumerate(physical):
                batch_entries = []
                for batch_id, indices in enumerate(batches):
                    relative = f"shards/{shard_id:03d}/batch-{batch_id:06d}.npz"
                    path = self._safe(workspace, relative)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    entry = BatchBuilder().write(path, samples, indices)
                    batch_entries.append(
                        {"batch_id": batch_id, "relative_filename": relative, **entry}
                    )
                shard = manifest_builder.shard(
                    config.dataset_build_id, shard_id, tuple(batch_entries)
                )
                relative_manifest = f"shards/{shard_id:03d}/shard-manifest.json"
                self._write(self._safe(workspace, relative_manifest), shard.content)
                shard_entries.append(
                    {
                        "shard_id": shard_id,
                        "sample_count": sum(entry["sample_count"] for entry in batch_entries),
                        "batch_count": len(batch_entries),
                        "relative_shard_manifest_path": relative_manifest,
                        "shard_manifest_sha256": shard.sha256,
                    }
                )
            root = manifest_builder.root(
                config, samples.sample_count, batch_counts[0], tuple(shard_entries)
            )
            self._write(workspace / "dataset-manifest.json", root.content)
            self._verify_tree(workspace, root.sha256)
            if final.exists():
                self._remove_owned_workspace(workspace)
                existing = self.load(config.dataset_build_id, expected_manifest_hash=root.sha256)
                if existing.dataset_manifest_hash != root.sha256:
                    raise ValueError("Incompatible build artifact already exists")
                self._verify_tree(final, root.sha256)
                return existing
            workspace.rename(final)
            self._fsync_directory(self._root)
            return PublishedDatasetBuild(
                config.dataset_build_id,
                root.sha256,
                final,
                final / "dataset-manifest.json",
            )
        except Exception:
            self._remove_owned_workspace(workspace)
            raise

    def load(
        self, dataset_build_id: str, expected_manifest_hash: str | None = None
    ) -> PublishedDatasetBuild:
        directory = self._root / self._directory_key(dataset_build_id)
        if not directory.is_dir():
            raise ValueError("Published Dataset Build is unavailable")
        manifest_path = directory / "dataset-manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Published Dataset Build is unavailable")
        manifest = DatasetManifest(manifest_path.read_bytes())
        if manifest.value.get("dataset_build_id") != dataset_build_id:
            raise ValueError("Dataset Build identity mismatch")
        if (
            expected_manifest_hash is not None
            and manifest.dataset_manifest_hash != expected_manifest_hash
        ):
            raise ValueError(
                f"Corrupted root manifest for build {dataset_build_id}: "
                f"expected {expected_manifest_hash}, got {manifest.dataset_manifest_hash}"
            )
        return PublishedDatasetBuild(
            dataset_build_id, manifest.dataset_manifest_hash, directory, manifest_path
        )

    def verify(self, published: PublishedDatasetBuild) -> DatasetManifest:
        expected = self._root / self._directory_key(published.dataset_build_id)
        if (
            published.directory.resolve() != expected
            or published.directory.is_symlink()
            or published.manifest_path != expected / "dataset-manifest.json"
        ):
            raise ValueError("Published paths do not belong to the Dataset Store")
        manifest = self._verify_tree(expected, published.dataset_manifest_hash)
        if manifest.value["dataset_build_id"] != published.dataset_build_id:
            raise ValueError("Dataset Build identity mismatch")
        return manifest

    def resolve_artifact(self, published: PublishedDatasetBuild, relative: str) -> Path:
        expected = self._root / self._directory_key(published.dataset_build_id)
        if (
            published.directory.resolve() != expected
            or published.directory.is_symlink()
            or published.manifest_path != expected / "dataset-manifest.json"
        ):
            raise ValueError("Published paths do not belong to the Dataset Store")
        path = self._safe(published.directory, relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Artifact is not a regular published file")
        return path

    def purge(self, dataset_build_id: str, dataset_manifest_hash: str) -> None:
        """Remove one verified immutable build selected by its full identity."""
        published = self.load(dataset_build_id, expected_manifest_hash=dataset_manifest_hash)
        if published.dataset_manifest_hash != dataset_manifest_hash:
            raise ValueError("Refusing to purge a Dataset Build with mismatched identity")
        self.verify(published)
        expected = self._root / self._directory_key(dataset_build_id)
        if published.directory.resolve() != expected or expected.parent != self._root:
            raise ValueError("Refusing to purge outside the Dataset Store")
        shutil.rmtree(expected)
        if expected.exists():
            raise OSError("Dataset Build artifact removal did not complete")
        self._fsync_directory(self._root)

    def persist_record(self, record_data: dict[str, Any]) -> None:
        """Persist durable metadata record to local store."""
        self._metadata.mkdir(parents=True, exist_ok=True)
        build_id = record_data["dataset_build_id"]
        destination = self._metadata / f"{build_id}.json"
        temporary = destination.with_suffix(f".{uuid4().hex}.tmp")
        content = canonical_json_bytes(record_data)
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)

    def load_records(self) -> list[dict[str, Any]]:
        """Load all durable metadata records from local store."""
        records = []
        if self._metadata.is_dir():
            for path in sorted(self._metadata.glob("*.json")):
                try:
                    records.append(json.loads(path.read_bytes()))
                except Exception as exc:
                    logger.warning("Failed to load local record %s: %s", path, exc)
        return records

    def probe_writable(self) -> bool:
        """Cheap local writability probe; never holds the service lock."""
        probe = self._root / f".health-probe-{uuid4().hex}"
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(b"")
            return True
        except OSError:
            return False
        finally:
            with contextlib.suppress(OSError):
                probe.unlink(missing_ok=True)

    def _verify_tree(self, directory: Path, expected_root_hash: str) -> DatasetManifest:
        manifest_path = self._safe(directory, "dataset-manifest.json")
        manifest = DatasetManifest(manifest_path.read_bytes())
        if manifest.dataset_manifest_hash != expected_root_hash:
            raise ValueError("Corrupt Root Dataset Manifest")
        root = manifest.value
        if set(root) != {
            "schema_version",
            "dataset_build_id",
            "dataset_id",
            "name",
            "profile",
            "task_type",
            "input_shape",
            "dtype",
            "num_classes",
            "preprocessing",
            "batch_size",
            "shard_count",
            "partition_algorithm",
            "partition_seed",
            "sample_count",
            "batch_count_per_shard",
            "shards",
        }:
            raise ValueError("Invalid Root Dataset Manifest fields")
        shards = root.get("shards")
        if not isinstance(shards, list) or len(shards) != root.get("shard_count"):
            raise ValueError("Invalid Root Dataset Manifest")
        seen: set[int] = set()
        total = 0
        for shard_id, reference in enumerate(shards):
            if (
                set(reference)
                != {
                    "shard_id",
                    "sample_count",
                    "batch_count",
                    "relative_shard_manifest_path",
                    "shard_manifest_sha256",
                }
                or reference.get("shard_id") != shard_id
            ):
                raise ValueError("Unordered Shard Manifest reference")
            shard_path = self._safe(directory, reference["relative_shard_manifest_path"])
            if sha256_file(str(shard_path)) != reference["shard_manifest_sha256"]:
                raise ValueError("Corrupt Shard Manifest")
            raw = shard_path.read_bytes()
            shard = json.loads(raw)
            if canonical_json_bytes(shard) != raw:
                raise ValueError("Noncanonical Shard Manifest")
            batches = shard.get("batches")
            if (
                set(shard)
                != {"dataset_build_id", "shard_id", "sample_count", "batch_count", "batches"}
                or shard.get("dataset_build_id") != root.get("dataset_build_id")
                or shard.get("shard_id") != shard_id
                or not isinstance(batches, list)
                or len(batches) != root.get("batch_count_per_shard")
                or shard.get("batch_count") != len(batches)
            ):
                raise ValueError("Inconsistent Shard Manifest")
            shard_total = 0
            for batch_id, entry in enumerate(batches):
                if set(entry) != {
                    "batch_id",
                    "relative_filename",
                    "sample_count",
                    "byte_size",
                    "sha256",
                }:
                    raise ValueError("Invalid Batch entry fields")
                path = self._safe(directory, entry["relative_filename"])
                if (
                    entry.get("batch_id") != batch_id
                    or path.stat().st_size != entry["byte_size"]
                    or sha256_file(str(path)) != entry["sha256"]
                ):
                    raise ValueError("Corrupt physical batch")
                with np.load(path, allow_pickle=False) as batch:
                    if set(batch.files) != {"x", "y", "sample_ids"}:
                        raise ValueError("Invalid NPZ fields")
                    x, y, ids = batch["x"], batch["y"], batch["sample_ids"]
                if (
                    x.dtype != np.float32
                    or x.ndim != 4
                    or tuple(x.shape[1:]) != tuple(root["input_shape"])
                    or y.dtype != np.int64
                    or ids.dtype != np.int64
                    or y.shape != (len(x),)
                    or ids.shape != (len(x),)
                    or len(x) != entry["sample_count"]
                    or not len(x)
                ):
                    raise ValueError("Invalid canonical NPZ")
                values = {int(value) for value in ids}
                if len(values) != len(ids) or seen.intersection(values):
                    raise ValueError("Duplicated sample identity")
                seen.update(values)
                shard_total += len(x)
            if shard_total != shard.get("sample_count") or shard_total != reference["sample_count"]:
                raise ValueError("Shard sample count mismatch")
            total += shard_total
        if total != root.get("sample_count") or seen != set(range(total)):
            raise ValueError("Dataset coverage mismatch")
        return manifest

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name != "nt":
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _remove_owned_workspace(self, workspace: Path) -> None:
        resolved = workspace.resolve()
        if resolved.parent != self._temporary_root.resolve():
            raise ValueError("Refusing to remove an unowned temporary directory")
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)


# Backwards compatibility alias
DatasetStorage = LocalArtifactStore


class HuggingFaceArtifactStore:
    """Hugging Face Dataset repository durable backend.

    Composes LocalArtifactStore for deterministic materialization, local staging, and read cache.
    Authoritative remote hierarchy matches the local layout exactly under
    dataset-builds/{build_id}/.
    """

    def __init__(
        self,
        local_root: Path,
        repo_id: str,
        token: str,
        branch: str = "main",
        api: Any | None = None,
    ):
        self._local_store = LocalArtifactStore(local_root)
        self._root = self._local_store._root
        self._temporary_root = self._local_store._temporary_root
        self._repo_id = repo_id
        self._token = token
        self._branch = branch
        self._cache_lock = threading.RLock()
        if api is not None:
            self._api = api
        else:
            from huggingface_hub import HfApi

            self._api = HfApi(token=token)
        self._verify_remote_branch()

    def _verify_remote_branch(self) -> None:
        """Verify that the configured branch exists on the remote Hugging Face repository."""
        try:
            refs = self._api.list_repo_refs(
                repo_id=self._repo_id,
                repo_type="dataset",
                token=self._token,
            )
        except Exception as exc:
            msg = (
                f"Failed to access Hugging Face repository '{self._repo_id}' "
                f"to verify branch '{self._branch}': {exc}"
            )
            raise RuntimeError(msg) from exc

        branches = [b.name for b in refs.branches]
        if self._branch not in branches:
            msg = (
                f"Configured branch '{self._branch}' does not exist in repository '{self._repo_id}'"
            )
            raise ValueError(msg)

    def materialize(
        self, config: DatasetBuildConfig, samples: SampleSource
    ) -> PublishedDatasetBuild:
        """Materialize locally, upload complete tree to Hugging Face, verify remote, cache."""
        # Compose LocalArtifactStore for deterministic local generation
        published = self._local_store.materialize(config, samples)
        remote_prefix = f"dataset-builds/{config.dataset_build_id}"

        # Upload folder to Hugging Face
        self._api.upload_folder(
            folder_path=str(published.directory),
            path_in_repo=remote_prefix,
            repo_id=self._repo_id,
            repo_type="dataset",
            revision=self._branch,
            token=self._token,
            commit_message=f"Publish artifacts for dataset build {config.dataset_build_id}",
        )

        # Remote verification: ensure uploaded files match byte size and sha256
        self._verify_remote_build(config.dataset_build_id, published.dataset_manifest_hash)
        return self._with_artifact_location(published)

    def load(
        self, dataset_build_id: str, expected_manifest_hash: str | None = None
    ) -> PublishedDatasetBuild:
        """Load root manifest from cache or remote Hugging Face repo with verification."""
        if expected_manifest_hash is None:
            raise ValueError("expected_manifest_hash is required for Hugging Face artifact loading")

        dir_key = self._local_store._directory_key(dataset_build_id)
        local_dir = self._root / dir_key
        manifest_path = local_dir / "dataset-manifest.json"

        with self._cache_lock:
            if manifest_path.is_file():
                raw = manifest_path.read_bytes()
                if sha256_bytes(raw) == expected_manifest_hash:
                    manifest = DatasetManifest(raw)
                    if manifest.value.get("dataset_build_id") == dataset_build_id:
                        return self._with_artifact_location(
                            PublishedDatasetBuild(
                                dataset_build_id, expected_manifest_hash, local_dir, manifest_path
                            )
                        )

            # Cache miss or invalid cache: fetch from remote Hugging Face repo.
            remote_path = f"dataset-builds/{dataset_build_id}/dataset-manifest.json"
            raw = self._download_remote_file(remote_path)
            if sha256_bytes(raw) != expected_manifest_hash:
                raise ValueError(
                    f"Corrupted root manifest for build {dataset_build_id} on remote: "
                    f"expected {expected_manifest_hash}, got {sha256_bytes(raw)}"
                )
            manifest = DatasetManifest(raw)
            if manifest.value.get("dataset_build_id") != dataset_build_id:
                raise ValueError("Dataset Build identity mismatch in remote manifest")

            self._atomic_cache_write(manifest_path, raw)
            return self._with_artifact_location(
                PublishedDatasetBuild(
                    dataset_build_id, expected_manifest_hash, local_dir, manifest_path
                )
            )

    def _with_artifact_location(self, published: PublishedDatasetBuild) -> PublishedDatasetBuild:
        repo = quote(self._repo_id, safe="/")
        revision = quote(self._branch, safe="")
        build = quote(published.dataset_build_id, safe="")
        base = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/dataset-builds/{build}"
        return PublishedDatasetBuild(
            published.dataset_build_id,
            published.dataset_manifest_hash,
            published.directory,
            published.manifest_path,
            base,
            "dataset-manifest.json",
        )

    def verify(self, published: PublishedDatasetBuild) -> DatasetManifest:
        """Verify build artifacts using local store verification rules."""
        return self._local_store.verify(published)

    def resolve_artifact(self, published: PublishedDatasetBuild, relative: str) -> Path:
        """Resolve artifact relative path with cache verification and remote refetch."""
        with self._cache_lock:
            return self._resolve_artifact_locked(published, relative)

    def _resolve_artifact_locked(self, published: PublishedDatasetBuild, relative: str) -> Path:
        local_path = self._local_store._safe(published.directory, relative)
        root_data = json.loads(published.manifest_path.read_bytes())

        expected_sha256: str | None = None
        expected_bytes: int | None = None
        is_shard = False

        for shard_ref in root_data.get("shards", []):
            if shard_ref.get("relative_shard_manifest_path") == relative:
                expected_sha256 = shard_ref.get("shard_manifest_sha256")
                is_shard = True
                break

        if not is_shard:
            batch_found = False
            for shard_ref in root_data.get("shards", []):
                shard_rel = shard_ref.get("relative_shard_manifest_path")
                shard_path = self._resolve_artifact_locked(published, shard_rel)
                shard_data = json.loads(shard_path.read_bytes())
                for batch_entry in shard_data.get("batches", []):
                    if batch_entry.get("relative_filename") == relative:
                        expected_bytes = batch_entry.get("byte_size")
                        expected_sha256 = batch_entry.get("sha256")
                        batch_found = True
                        break
                if batch_found:
                    break
            if not batch_found:
                raise ValueError(f"Unknown artifact path cannot be verified: {relative}")

        # Cache hit check: verify against manifest chain; refetch if corrupted
        if local_path.is_file():
            valid = True
            if (expected_bytes is not None and local_path.stat().st_size != expected_bytes) or (
                expected_sha256 is not None and sha256_file(str(local_path)) != expected_sha256
            ):
                valid = False

            if valid:
                return local_path
            local_path.unlink(missing_ok=True)

        # Cache miss (or invalidated cache hit): download from remote Hugging Face repository
        remote_path = f"dataset-builds/{published.dataset_build_id}/{relative}"
        content = self._download_remote_file(remote_path)

        if expected_bytes is not None and len(content) != expected_bytes:
            raise ValueError(
                f"Remote Batch byte size mismatch for {relative}: "
                f"expected {expected_bytes}, got {len(content)}"
            )
        if expected_sha256 is not None and sha256_bytes(content) != expected_sha256:
            desc = "Shard Manifest" if is_shard else "Batch"
            raise ValueError(
                f"Remote {desc} SHA-256 mismatch for {relative}: "
                f"expected {expected_sha256}, got {sha256_bytes(content)}"
            )

        self._atomic_cache_write(local_path, content)
        return local_path

    @staticmethod
    def _atomic_cache_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def purge(self, dataset_build_id: str, dataset_manifest_hash: str) -> None:
        """Purge remote artifact folder and local cache for the specified build.

        Idempotent for crash recovery: if the remote folder is confirmed to be
        already absent, this succeeds rather than failing startup or retry.
        Network, authentication, and remote server errors fail closed.
        """
        remote_prefix = f"dataset-builds/{dataset_build_id}"

        # 1. Check if remote folder is already absent
        try:
            remote_files = self._api.list_repo_files(
                repo_id=self._repo_id,
                repo_type="dataset",
                revision=self._branch,
                token=self._token,
            )
        except Exception as exc:
            msg = (
                f"Failed to inspect remote repository '{self._repo_id}' "
                f"during purge of '{dataset_build_id}': {exc}"
            )
            raise RuntimeError(msg) from exc

        has_remote_folder = any(
            f == remote_prefix or f.startswith(f"{remote_prefix}/") for f in remote_files
        )

        if has_remote_folder:
            try:
                self._api.delete_folder(
                    path_in_repo=remote_prefix,
                    repo_id=self._repo_id,
                    repo_type="dataset",
                    revision=self._branch,
                    token=self._token,
                    commit_message=f"Purge artifacts for dataset build {dataset_build_id}",
                )
            except Exception as exc:
                from huggingface_hub.utils import EntryNotFoundError

                if isinstance(exc, EntryNotFoundError):
                    pass
                else:
                    raise

        # 2. Local cache cleanup (if any partial files remain locally)
        dir_key = self._local_store._directory_key(dataset_build_id)
        local_dir = self._root / dir_key
        if local_dir.exists():
            shutil.rmtree(local_dir, ignore_errors=True)

    def persist_record(self, record_data: dict[str, Any]) -> None:
        """Persist durable metadata record to local cache and remote Hugging Face repository."""
        build_id = record_data["dataset_build_id"]
        # Save locally in cache
        self._local_store.persist_record(record_data)

        # Upload to remote Hugging Face repo
        remote_path = f".service/builds/{build_id}.json"
        content = canonical_json_bytes(record_data)
        self._api.upload_file(
            path_or_fileobj=content,
            path_in_repo=remote_path,
            repo_id=self._repo_id,
            repo_type="dataset",
            revision=self._branch,
            token=self._token,
            commit_message=f"Persist metadata for dataset build {build_id}",
        )

    def load_records(self) -> list[dict[str, Any]]:
        """Load all durable metadata records from remote repository, syncing to local cache."""
        try:
            files = self._api.list_repo_files(
                repo_id=self._repo_id,
                repo_type="dataset",
                revision=self._branch,
                token=self._token,
            )
        except Exception as exc:
            msg = (
                f"Failed to list remote records from Hugging Face repository "
                f"'{self._repo_id}': {exc}"
            )
            raise RuntimeError(msg) from exc

        remote_metadata_files = {
            Path(f).name: f
            for f in files
            if f.startswith(".service/builds/") and f.endswith(".json")
        }

        # Purge stale local cache files that no longer exist on remote
        if self._local_store._metadata.is_dir():
            for local_file in self._local_store._metadata.glob("*.json"):
                if local_file.name not in remote_metadata_files:
                    local_file.unlink(missing_ok=True)

        records: list[dict[str, Any]] = []
        for filename, remote_file in sorted(remote_metadata_files.items()):
            try:
                content = self._download_remote_file(remote_file)
                local_dest = self._local_store._metadata / filename
                local_dest.parent.mkdir(parents=True, exist_ok=True)
                local_dest.write_bytes(content)
                records.append(json.loads(content))
            except Exception as exc:
                msg = f"Failed to download remote metadata record '{remote_file}': {exc}"
                raise RuntimeError(msg) from exc

        return records

    def probe_writable(self) -> bool:
        """Non-mutating permission probe verifying token validity and repo write permission."""
        try:
            self._api.auth_check(
                repo_id=self._repo_id,
                repo_type="dataset",
                token=self._token,
                write=True,
            )
            return True
        except Exception:
            return False

    def _download_remote_file(self, path_in_repo: str) -> bytes:
        if hasattr(self._api, "download_file"):
            return self._api.download_file(
                repo_id=self._repo_id,
                filename=path_in_repo,
                revision=self._branch,
            )
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=self._repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            revision=self._branch,
            token=self._token,
        )
        return Path(local_path).read_bytes()

    def _verify_remote_build(self, dataset_build_id: str, expected_manifest_hash: str) -> None:
        """Verify all remote files exist and match exact SHA-256 and byte size."""
        root_content = self._download_remote_file(
            f"dataset-builds/{dataset_build_id}/dataset-manifest.json"
        )
        if sha256_bytes(root_content) != expected_manifest_hash:
            raise ValueError("Remote Root Dataset Manifest hash mismatch")
        root = json.loads(root_content)

        for shard_ref in root.get("shards", []):
            rel_shard = shard_ref["relative_shard_manifest_path"]
            shard_content = self._download_remote_file(
                f"dataset-builds/{dataset_build_id}/{rel_shard}"
            )
            if sha256_bytes(shard_content) != shard_ref["shard_manifest_sha256"]:
                raise ValueError(f"Remote Shard Manifest hash mismatch: {rel_shard}")
            shard = json.loads(shard_content)

            for batch_entry in shard.get("batches", []):
                rel_batch = batch_entry["relative_filename"]
                batch_content = self._download_remote_file(
                    f"dataset-builds/{dataset_build_id}/{rel_batch}"
                )
                if len(batch_content) != batch_entry["byte_size"]:
                    raise ValueError(f"Remote Batch byte size mismatch: {rel_batch}")
                if sha256_bytes(batch_content) != batch_entry["sha256"]:
                    raise ValueError(f"Remote Batch SHA-256 mismatch: {rel_batch}")
