"""Persistent single-worker Dataset Build orchestration and artifact state gates."""

import contextlib
import hashlib
import json
import os
import shutil
import tarfile
import threading
import urllib.request
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from dataset_manager.config import DatasetBuildConfig, DatasetManagerConfig
from dataset_manager.hashing import canonical_json_bytes, sha256_bytes
from dataset_manager.importer import DatasetImporter
from dataset_manager.preprocessing import Preprocessor
from dataset_manager.schemas import CreateBuildRequest, DatasetBuildState
from dataset_manager.storage import DatasetStorage, PublishedDatasetBuild

_CIFAR10_BINARY_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
_CIFAR10_BINARY_ARCHIVE_BYTES = 170_052_171
_CIFAR10_BINARY_ARCHIVE_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
_ACTIVE_BUILD_STATES = {
    DatasetBuildState.IMPORTING,
    DatasetBuildState.VALIDATING,
    DatasetBuildState.PREPROCESSING,
    DatasetBuildState.MATERIALIZING,
    DatasetBuildState.VERIFYING,
}


class DatasetServiceError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        *,
        retryable: bool = False,
        details: dict[str, object] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class BuildExecutionResult:
    published: PublishedDatasetBuild
    raw_workspace: Path | None = None


@dataclass(slots=True)
class _BuildRecord:
    dataset_build_id: str
    request: dict[str, Any]
    request_fingerprint: str
    idempotency_key: str
    state: str
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    current_stage: str | None = None
    progress: float | None = None
    dataset_manifest_hash: str | None = None
    manifest_uri: str | None = None
    artifact_base_url: str | None = None
    sample_count: int | None = None
    registration_id: str | None = None
    registration_acknowledged_at: str | None = None
    raw_workspace: str | None = None
    error: dict[str, object] | None = None
    purge_command_id: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DatasetBuildPipeline:
    """Allowlisted CIFAR-10 source acquisition and deterministic artifact build."""

    def __init__(self, config: DatasetManagerConfig, storage: DatasetStorage):
        self._config = config
        self._storage = storage

    def __call__(
        self,
        dataset_build_id: str,
        request: dict[str, Any],
        update: Callable[[DatasetBuildState, str, float], None],
    ) -> BuildExecutionResult:
        workspace = Path(self._config.temp_dir).resolve() / dataset_build_id
        if workspace.exists():
            raise ValueError("Build source workspace already exists")
        workspace.mkdir(parents=True)
        archive_part = workspace / "cifar-10-binary.tar.gz.part"
        archive = workspace / "cifar-10-binary.tar.gz"
        update(DatasetBuildState.IMPORTING, "DOWNLOADING", 0.05)
        self._download(archive_part)
        os.replace(archive_part, archive)
        extracted = workspace / "source"
        extracted.mkdir()
        self._extract(archive, extracted)
        files = tuple(
            extracted / "cifar-10-batches-bin" / f"data_batch_{index}.bin" for index in range(1, 6)
        )
        imported = DatasetImporter().import_cifar10_binary(files)
        update(DatasetBuildState.VALIDATING, "VALIDATING_SOURCE", 0.25)
        if request["source"]["version"] != "binary-v1":
            raise ValueError("Unsupported CIFAR-10 source version")
        update(DatasetBuildState.PREPROCESSING, "STATIC_PREPROCESSING", 0.4)
        normalization = request["normalization"]
        samples = Preprocessor(
            tuple(request["input_shape"]),
            10,
            tuple(normalization["mean"]),
            tuple(normalization["std"]),
        ).transform(imported.x, imported.y)
        build_config = DatasetBuildConfig(
            1,
            dataset_build_id,
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
                "mean": normalization["mean"],
                "std": normalization["std"],
            },
            request["batch_size"],
            request["shard_count"],
            "seeded_permutation_round_robin",
            request["partition_seed"],
        )
        update(DatasetBuildState.MATERIALIZING, "MATERIALIZING_ARTIFACTS", 0.55)
        published = self._storage.materialize(build_config, samples)
        update(DatasetBuildState.VERIFYING, "VERIFYING_HASH_CHAIN", 0.9)
        self._storage.verify(published)
        return BuildExecutionResult(published, workspace)

    def _download(self, destination: Path) -> None:
        cache_env = os.environ.get("PBL4_CIFAR10_SOURCE_ARCHIVE")
        cache_candidates = [
            Path(cache_env).resolve() if cache_env else None,
            (Path.cwd() / ".var" / "cache" / "cifar-10-binary.tar.gz").resolve(),
        ]
        for candidate in cache_candidates:
            if (
                candidate
                and candidate.is_file()
                and candidate.stat().st_size == _CIFAR10_BINARY_ARCHIVE_BYTES
            ):
                with candidate.open("rb") as source:
                    digest = hashlib.file_digest(source, "md5").hexdigest()
                if digest == _CIFAR10_BINARY_ARCHIVE_MD5:
                    shutil.copyfile(candidate, destination)
                    return

        request = urllib.request.Request(
            _CIFAR10_BINARY_URL, headers={"User-Agent": "pbl4/1"}, method="HEAD"
        )
        with urllib.request.urlopen(
            request, timeout=self._config.source_download_timeout_seconds
        ) as response:
            content_length = int(response.headers.get("Content-Length", "0"))
            supports_ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
            etag = response.headers.get("ETag")
            resolved_url = response.geturl()
        if content_length != _CIFAR10_BINARY_ARCHIVE_BYTES:
            raise ValueError("Official CIFAR-10 source has an unexpected archive size")
        if not supports_ranges or self._config.download_parallelism == 1:
            self._download_sequential(destination)
            if destination.stat().st_size != content_length:
                raise ValueError("Incomplete CIFAR-10 source download")
            return

        parallelism = min(self._config.download_parallelism, content_length)
        span = (content_length + parallelism - 1) // parallelism
        ranges = tuple(
            (index, index * span, min(content_length - 1, (index + 1) * span - 1))
            for index in range(parallelism)
            if index * span < content_length
        )
        parts = tuple(
            destination.with_name(f"{destination.name}.{index}.range") for index, _, _ in ranges
        )
        try:
            with ThreadPoolExecutor(
                max_workers=len(ranges), thread_name_prefix="cifar10-range"
            ) as pool:
                futures = [
                    pool.submit(
                        self._download_range,
                        resolved_url,
                        part,
                        start,
                        end,
                        content_length,
                        etag,
                    )
                    for part, (_, start, end) in zip(parts, ranges, strict=True)
                ]
                for future in futures:
                    future.result()
            with destination.open("xb") as output:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, output, self._config.download_chunk_size)
                output.flush()
                os.fsync(output.fileno())
            if destination.stat().st_size != content_length:
                raise ValueError("Incomplete parallel CIFAR-10 source download")
        finally:
            for part in parts:
                with contextlib.suppress(FileNotFoundError):
                    part.unlink()

    def _download_sequential(self, destination: Path) -> None:
        total = 0
        request = urllib.request.Request(_CIFAR10_BINARY_URL, headers={"User-Agent": "pbl4/1"})
        with (
            urllib.request.urlopen(
                request, timeout=self._config.source_download_timeout_seconds
            ) as response,
            destination.open("xb") as stream,
        ):
            while chunk := response.read(self._config.download_chunk_size):
                total += len(chunk)
                if total > self._config.max_source_bytes:
                    raise ValueError("CIFAR-10 source exceeds configured size limit")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if not total:
            raise ValueError("Empty CIFAR-10 source download")

    def _download_range(
        self,
        source_url: str,
        destination: Path,
        start: int,
        end: int,
        total_size: int,
        etag: str | None,
    ) -> None:
        headers = {"User-Agent": "pbl4/1", "Range": f"bytes={start}-{end}"}
        if etag:
            headers["If-Range"] = etag
        request = urllib.request.Request(source_url, headers=headers)
        expected = end - start + 1
        received = 0
        with (
            urllib.request.urlopen(
                request, timeout=self._config.source_download_timeout_seconds
            ) as response,
            destination.open("xb") as stream,
        ):
            if response.status != 206 or response.headers.get("Content-Range") != (
                f"bytes {start}-{end}/{total_size}"
            ):
                raise ValueError("Official CIFAR-10 server returned an invalid byte range")
            if etag and response.headers.get("ETag") != etag:
                raise ValueError("Official CIFAR-10 source identity changed during download")
            read_size = min(self._config.download_chunk_size, 64 * 1024)
            while chunk := response.read(read_size):
                received += len(chunk)
                if received > expected:
                    raise ValueError("CIFAR-10 range exceeded its requested extent")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if received != expected:
            raise ValueError("Incomplete CIFAR-10 byte range")

    def _extract(self, archive: Path, destination: Path) -> None:
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            if any(not member.isfile() and not member.isdir() for member in members):
                raise ValueError("Unsupported CIFAR-10 archive member")
            total = sum(member.size for member in members)
            if total > self._config.max_source_bytes:
                raise ValueError("Expanded CIFAR-10 source exceeds configured size limit")
            root = destination.resolve()
            for member in members:
                candidate = (destination / member.name).resolve()
                if candidate != root and root not in candidate.parents:
                    raise ValueError("CIFAR-10 archive path traversal")
            bundle.extractall(destination, filter="data")


class DatasetService:
    """Owns Dataset Build state, bounded FIFO execution, registration, and artifacts."""

    def __init__(
        self,
        config: DatasetManagerConfig,
        executor: Callable[
            [str, dict[str, Any], Callable[[DatasetBuildState, str, float], None]],
            BuildExecutionResult,
        ]
        | None = None,
        *,
        start_worker: bool = True,
    ):
        self.config = config
        self.storage = DatasetStorage(Path(config.store_dir))
        self._metadata = Path(config.store_dir).resolve() / ".service" / "builds"
        self._metadata.mkdir(parents=True, exist_ok=True)
        self._executor = executor or DatasetBuildPipeline(config, self.storage)
        self._condition = threading.Condition()
        self._records: dict[str, _BuildRecord] = {}
        self._idempotency: dict[str, str] = {}
        self._queue: deque[str] = deque()
        self._active: str | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        self._load()
        if start_worker:
            self.start()

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run, name="dataset-build-worker", daemon=True
            )
            self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def health(self) -> dict[str, object]:
        """Return a thread-safe snapshot of local Dataset Manager health.

        Health reflects local service capability only.  Backend/PostgreSQL
        reachability and REGISTERING state do NOT affect health status.
        """
        with self._condition:
            queue_depth = len(self._queue)
            active_build_id = self._active
        storage_writable = self._probe_storage_writable()
        status = "ok" if storage_writable else "degraded"
        return {
            "status": status,
            "service": "dataset-manager",
            "version": "1",
            "queue_depth": queue_depth,
            "active_build_id": active_build_id,
            "storage_writable": storage_writable,
        }

    def _probe_storage_writable(self) -> bool:
        """Cheap local writability probe; never holds the service lock."""
        store = Path(self.config.store_dir)
        probe = store / f".health-probe-{uuid4().hex}"
        try:
            store.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(b"")
            return True
        except OSError:
            return False
        finally:
            with contextlib.suppress(OSError):
                probe.unlink(missing_ok=True)

    def submit(self, request: dict[str, Any], idempotency_key: str) -> dict[str, object]:
        if not idempotency_key:
            raise DatasetServiceError("INVALID_REQUEST", "Missing Idempotency-Key", 400)
        frozen_request = json.loads(canonical_json_bytes(request))
        fingerprint = sha256_bytes(canonical_json_bytes(frozen_request))
        with self._condition:
            if existing_id := self._idempotency.get(idempotency_key):
                existing = self._records[existing_id]
                if existing.request_fingerprint != fingerprint:
                    raise DatasetServiceError(
                        "IDEMPOTENCY_CONFLICT",
                        "Idempotency-Key was reused with a different request",
                        409,
                    )
                return self._submission(existing)
            if len(self._queue) >= self.config.queue_capacity:
                raise DatasetServiceError(
                    "BUILD_QUEUE_FULL",
                    "Dataset Build queue is full",
                    429,
                    retryable=True,
                )
            dataset_build_id = str(uuid4())
            now = _now()
            record = _BuildRecord(
                dataset_build_id,
                frozen_request,
                fingerprint,
                idempotency_key,
                DatasetBuildState.CREATED,
                now,
                now,
            )
            self._records[dataset_build_id] = record
            self._idempotency[idempotency_key] = dataset_build_id
            self._queue.append(dataset_build_id)
            if self._active is not None or len(self._queue) > 1:
                record.state = DatasetBuildState.QUEUED
            self._persist(record)
            self._condition.notify()
            return self._submission(record)

    def rebuild(
        self,
        dataset_build_id: str,
        overrides: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, object]:
        source = self._get(dataset_build_id)
        request = json.loads(canonical_json_bytes(source.request))
        for field in ("batch_size", "partition_seed", "normalization", "input_shape"):
            if overrides.get(field) is not None:
                request[field] = overrides[field]
        request["command_id"] = overrides["command_id"]
        try:
            request = CreateBuildRequest.model_validate(request).model_dump(mode="json")
        except ValidationError as exc:
            raise DatasetServiceError(
                "INVALID_REQUEST", "Rebuild overrides violate the dataset profile", 400
            ) from exc
        result = self.submit(request, idempotency_key)
        return {
            "source_dataset_build_id": dataset_build_id,
            "new_dataset_build_id": result["dataset_build_id"],
            "state": result["state"],
            "status_url": result["status_url"],
        }

    def status(self, dataset_build_id: str) -> dict[str, object]:
        with self._condition:
            record = self._get(dataset_build_id)
            value = self._status(record)
            value["queue_position"] = (
                list(self._queue).index(dataset_build_id) + 1
                if dataset_build_id in self._queue
                else None
            )
            return value

    def acknowledge_registration(
        self,
        dataset_build_id: str,
        dataset_manifest_hash: str,
        registration_id: str,
        catalog_persisted_at: str,
    ) -> dict[str, object]:
        del catalog_persisted_at
        with self._condition:
            record = self._get(dataset_build_id)
            if record.state == DatasetBuildState.READY:
                if (
                    record.registration_id == registration_id
                    and record.dataset_manifest_hash == dataset_manifest_hash
                ):
                    return self._status(record)
                raise DatasetServiceError(
                    "REGISTRATION_CONFLICT", "Registration ACK conflicts with READY build", 409
                )
            if record.state != DatasetBuildState.REGISTERING:
                raise DatasetServiceError(
                    "INVALID_STATE_TRANSITION", "Build is not REGISTERING", 409
                )
            if record.dataset_manifest_hash != dataset_manifest_hash:
                raise DatasetServiceError(
                    "REGISTRATION_CONFLICT", "Dataset Manifest hash does not match", 409
                )
            record.registration_id = registration_id
            record.registration_acknowledged_at = _now()
            record.state = DatasetBuildState.READY
            record.updated_at = record.registration_acknowledged_at
            self._persist(record)
            result = self._status(record)
        self._cleanup_raw(record)
        return result

    def deprecate(self, dataset_build_id: str) -> dict[str, object]:
        with self._condition:
            record = self._get(dataset_build_id)
            if record.state == DatasetBuildState.DEPRECATED:
                return self._status(record)
            if record.state != DatasetBuildState.READY:
                raise DatasetServiceError(
                    "INVALID_STATE_TRANSITION", "Only READY builds can be deprecated", 409
                )
            record.state = DatasetBuildState.DEPRECATED
            record.updated_at = _now()
            self._persist(record)
            return self._status(record)

    def purge(self, dataset_build_id: str, command_id: str) -> dict[str, object]:
        with self._condition:
            record = self._get(dataset_build_id)
            if record.state == DatasetBuildState.DELETED and record.purge_command_id == command_id:
                return self._status(record)
            if record.purge_command_id is not None and record.purge_command_id != command_id:
                raise DatasetServiceError(
                    "IDEMPOTENCY_CONFLICT", "Purge command conflicts with prior command", 409
                )
            if record.state not in (DatasetBuildState.DEPRECATED, DatasetBuildState.FAILED):
                raise DatasetServiceError(
                    "INVALID_STATE_TRANSITION",
                    "Only DEPRECATED or FAILED builds can be purged",
                    409,
                )
            record.state = DatasetBuildState.DELETING
            record.purge_command_id = command_id
            record.updated_at = _now()
            self._persist(record)
            manifest_hash = record.dataset_manifest_hash
        if manifest_hash is not None:
            self.storage.purge(dataset_build_id, manifest_hash)
        self._cleanup_raw(record)
        with self._condition:
            record.state = DatasetBuildState.DELETED
            record.updated_at = _now()
            self._persist(record)
            return self._status(record)

    def artifact(
        self, dataset_build_id: str, shard_id: int | None = None, batch_id: int | None = None
    ) -> tuple[Path, str, str]:
        with self._condition:
            record = self._get(dataset_build_id)
            allowed = (
                {
                    DatasetBuildState.REGISTERING,
                    DatasetBuildState.READY,
                    DatasetBuildState.DEPRECATED,
                }
                if shard_id is None
                else {DatasetBuildState.READY, DatasetBuildState.DEPRECATED}
            )
            if record.state not in allowed:
                raise DatasetServiceError("BUILD_NOT_READY", "Artifact is not visible", 409)
        published = self.storage.load(dataset_build_id)
        if shard_id is None:
            path = published.manifest_path
            digest = published.dataset_manifest_hash
            media_type = "application/json"
        else:
            root = json.loads(published.manifest_path.read_bytes())
            if shard_id < 0 or shard_id >= len(root["shards"]):
                raise DatasetServiceError("ARTIFACT_NOT_FOUND", "Shard is absent", 404)
            reference = root["shards"][shard_id]
            shard_path = self.storage.resolve_artifact(
                published, reference["relative_shard_manifest_path"]
            )
            if batch_id is None:
                path = shard_path
                digest = reference["shard_manifest_sha256"]
                media_type = "application/json"
            else:
                shard = json.loads(shard_path.read_bytes())
                if batch_id < 0 or batch_id >= len(shard["batches"]):
                    raise DatasetServiceError("ARTIFACT_NOT_FOUND", "Batch is absent", 404)
                entry = shard["batches"][batch_id]
                path = self.storage.resolve_artifact(published, entry["relative_filename"])
                digest = entry["sha256"]
                media_type = "application/octet-stream"
        return path, digest, media_type

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                dataset_build_id = self._queue.popleft()
                self._active = dataset_build_id
                record = self._records[dataset_build_id]
                record.started_at = record.started_at or _now()
                record.raw_workspace = str(
                    (Path(self.config.temp_dir).resolve() / dataset_build_id).resolve()
                )
                self._persist(record)
            try:
                result = self._executor(
                    dataset_build_id,
                    record.request,
                    lambda state, stage, progress, build_id=dataset_build_id: self._update(
                        build_id, state, stage, progress
                    ),
                )
                manifest = self.storage.verify(result.published).value
                with self._condition:
                    record.dataset_manifest_hash = result.published.dataset_manifest_hash
                    record.artifact_base_url = (
                        f"{self.config.public_base_url.rstrip('/')}"
                        f"/artifacts/v1/dataset-builds/{dataset_build_id}"
                    )
                    record.manifest_uri = f"{record.artifact_base_url}/manifest.json"
                    record.sample_count = manifest["sample_count"]
                    if result.raw_workspace is not None:
                        workspace = result.raw_workspace.resolve()
                        if workspace.parent != Path(self.config.temp_dir).resolve():
                            raise ValueError("Build returned an unowned source workspace")
                        record.raw_workspace = str(workspace)
                    record.state = DatasetBuildState.REGISTERING
                    record.current_stage = "AWAITING_REGISTRATION_ACK"
                    record.progress = 1.0
                    record.completed_at = _now()
                    record.updated_at = record.completed_at
                    self._persist(record)
            except Exception as exc:
                with self._condition:
                    record.state = DatasetBuildState.FAILED
                    record.error = {
                        "code": "BUILD_FAILED",
                        "stage": record.current_stage,
                        "message": str(exc),
                    }
                    record.completed_at = _now()
                    record.updated_at = record.completed_at
                    self._persist(record)
            finally:
                with self._condition:
                    self._active = None
                    self._condition.notify_all()

    def _update(
        self, dataset_build_id: str, state: DatasetBuildState, stage: str, progress: float
    ) -> None:
        with self._condition:
            record = self._records[dataset_build_id]
            if state not in _ACTIVE_BUILD_STATES or not 0 <= progress <= 1:
                raise ValueError("Invalid active Dataset Build update")
            record.state = state
            record.current_stage = stage
            record.progress = progress
            record.updated_at = _now()
            self._persist(record)

    def _load(self) -> None:
        queued: list[_BuildRecord] = []
        deleting: list[_BuildRecord] = []
        for path in self._metadata.glob("*.json"):
            value = json.loads(path.read_bytes())
            record = _BuildRecord(**value)
            self._records[record.dataset_build_id] = record
            self._idempotency[record.idempotency_key] = record.dataset_build_id
            state = DatasetBuildState(record.state)
            if state in (DatasetBuildState.CREATED, DatasetBuildState.QUEUED):
                record.state = DatasetBuildState.QUEUED
                queued.append(record)
            elif state in _ACTIVE_BUILD_STATES:
                record.state = DatasetBuildState.FAILED
                record.error = {
                    "code": "BUILD_INTERRUPTED",
                    "stage": record.current_stage,
                    "message": "Dataset Manager restarted during an active build",
                }
                record.updated_at = _now()
                record.completed_at = record.updated_at
                self._persist(record)
            elif state == DatasetBuildState.DELETING:
                deleting.append(record)
        for record in sorted(queued, key=lambda item: item.created_at):
            self._queue.append(record.dataset_build_id)
            self._persist(record)
        for record in deleting:
            try:
                if record.dataset_manifest_hash is not None:
                    self.storage.purge(record.dataset_build_id, record.dataset_manifest_hash)
            except ValueError:
                # A missing tree means the prior physical removal completed before
                # the DELETED metadata write.
                try:
                    self.storage.load(record.dataset_build_id)
                except ValueError:
                    pass
                else:
                    raise
            self._cleanup_raw(record)
            record.state = DatasetBuildState.DELETED
            record.updated_at = _now()
            self._persist(record)

    def _persist(self, record: _BuildRecord) -> None:
        destination = self._metadata / f"{record.dataset_build_id}.json"
        temporary = destination.with_suffix(f".{uuid4().hex}.tmp")
        content = canonical_json_bytes(asdict(record))
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)

    def _cleanup_raw(self, record: _BuildRecord) -> None:
        if not record.raw_workspace:
            return
        workspace = Path(record.raw_workspace).resolve()
        root = Path(self.config.temp_dir).resolve()
        if workspace.parent != root:
            raise ValueError("Refusing to clean an unowned source workspace")
        if workspace.exists():
            shutil.rmtree(workspace)
        with self._condition:
            if record.raw_workspace == str(workspace):
                record.raw_workspace = None
                self._persist(record)

    def _get(self, dataset_build_id: str) -> _BuildRecord:
        try:
            return self._records[dataset_build_id]
        except KeyError as exc:
            raise DatasetServiceError(
                "BUILD_NOT_FOUND", "Dataset Build was not found", 404
            ) from exc

    def _submission(self, record: _BuildRecord) -> dict[str, object]:
        return {
            "dataset_build_id": record.dataset_build_id,
            "state": record.state,
            "queue_position": (
                list(self._queue).index(record.dataset_build_id) + 1
                if record.dataset_build_id in self._queue
                else None
            ),
            "status_url": f"/api/v1/dataset-builds/{record.dataset_build_id}",
            "created_at": record.created_at,
        }

    def _status(self, record: _BuildRecord) -> dict[str, object]:
        request = record.request
        return {
            "dataset_build_id": record.dataset_build_id,
            "dataset_id": "cifar10",
            "name": "CIFAR-10",
            "state": record.state,
            "current_stage": record.current_stage,
            "progress": record.progress,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "updated_at": record.updated_at,
            "completed_at": record.completed_at,
            "dataset_manifest_hash": record.dataset_manifest_hash,
            "manifest_uri": record.manifest_uri,
            "artifact_base_url": record.artifact_base_url,
            "sample_count": record.sample_count,
            "shard_count": request["shard_count"],
            "batch_size": request["batch_size"],
            "profile": request["profile"],
            "task_type": "image_classification",
            "error": record.error,
            "registration_id": record.registration_id,
            "registration_acknowledged_at": record.registration_acknowledged_at,
        }
