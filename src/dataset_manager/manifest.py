"""Canonical Dataset and Shard Manifest construction and strict parsing."""

import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from dataset_manager.config import DatasetBuildConfig
from dataset_manager.hashing import canonical_json_bytes, sha256_bytes


def _thaw_json(value: Any) -> Any:
    if hasattr(value, "items"):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError("Unsafe manifest-relative path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class ManifestArtifact:
    content: bytes
    sha256: str

    @classmethod
    def create(cls, value: dict[str, object]) -> "ManifestArtifact":
        content = canonical_json_bytes(value)
        return cls(content, sha256_bytes(content))

    @property
    def value(self) -> dict[str, Any]:
        value = json.loads(self.content)
        if not isinstance(value, dict):
            raise ValueError("Manifest root must be an object")
        return value


class ManifestBuilder:
    def shard(
        self,
        dataset_build_id: str,
        shard_id: int,
        batches: tuple[dict[str, object], ...],
    ) -> ManifestArtifact:
        ordered = tuple(dict(batch) for batch in batches)
        for expected_id, batch in enumerate(ordered):
            if set(batch) != {
                "batch_id",
                "relative_filename",
                "sample_count",
                "byte_size",
                "sha256",
            }:
                raise ValueError("Invalid Batch entry fields")
            if (
                batch["batch_id"] != expected_id
                or type(batch["sample_count"]) is not int
                or batch["sample_count"] <= 0
                or type(batch["byte_size"]) is not int
                or batch["byte_size"] <= 0
                or not isinstance(batch["sha256"], str)
                or len(batch["sha256"]) != 64
            ):
                raise ValueError("Invalid Batch entry")
            batch["relative_filename"] = _relative_path(str(batch["relative_filename"]))
        return ManifestArtifact.create(
            {
                "dataset_build_id": dataset_build_id,
                "shard_id": shard_id,
                "sample_count": sum(int(batch["sample_count"]) for batch in ordered),
                "batch_count": len(ordered),
                "batches": list(ordered),
            }
        )

    def root(
        self,
        config: DatasetBuildConfig,
        sample_count: int,
        batch_count_per_shard: int,
        shards: tuple[dict[str, object], ...],
    ) -> ManifestArtifact:
        ordered = tuple(dict(shard) for shard in shards)
        if (
            type(sample_count) is not int
            or sample_count <= 0
            or type(batch_count_per_shard) is not int
            or batch_count_per_shard <= 0
            or len(ordered) != config.shard_count
        ):
            raise ValueError("Invalid root manifest counts")
        for expected_id, shard in enumerate(ordered):
            if set(shard) != {
                "shard_id",
                "sample_count",
                "batch_count",
                "relative_shard_manifest_path",
                "shard_manifest_sha256",
            }:
                raise ValueError("Invalid Shard entry fields")
            if (
                shard["shard_id"] != expected_id
                or shard["batch_count"] != batch_count_per_shard
                or type(shard["sample_count"]) is not int
                or shard["sample_count"] <= 0
                or not isinstance(shard["shard_manifest_sha256"], str)
                or len(shard["shard_manifest_sha256"]) != 64
            ):
                raise ValueError("Invalid Shard entry")
            shard["relative_shard_manifest_path"] = _relative_path(
                str(shard["relative_shard_manifest_path"])
            )
        if sum(int(shard["sample_count"]) for shard in ordered) != sample_count:
            raise ValueError("Shard sample counts do not cover the dataset")
        return ManifestArtifact.create(
            {
                "schema_version": config.schema_version,
                "dataset_build_id": config.dataset_build_id,
                "dataset_id": config.dataset_id,
                "name": config.name,
                "profile": config.profile,
                "task_type": config.task_type,
                "input_shape": list(config.input_shape),
                "dtype": config.dtype,
                "num_classes": config.num_classes,
                "preprocessing": _thaw_json(config.preprocessing),
                "batch_size": config.batch_size,
                "shard_count": config.shard_count,
                "partition_algorithm": config.partition_algorithm,
                "partition_seed": config.partition_seed,
                "sample_count": sample_count,
                "batch_count_per_shard": batch_count_per_shard,
                "shards": list(ordered),
            }
        )


class DatasetManifest:
    """Verified canonical root bytes and their dataset_manifest_hash."""

    def __init__(self, content: bytes):
        self._content = bytes(content)
        value = json.loads(self._content)
        if not isinstance(value, dict):
            raise ValueError("Dataset Manifest must be an object")
        if canonical_json_bytes(value) != self._content:
            raise ValueError("Dataset Manifest is not canonical JSON")
        self._value = value
        self.dataset_manifest_hash = sha256_bytes(self._content)

    @property
    def content(self) -> bytes:
        return self._content

    @property
    def value(self) -> dict[str, Any]:
        return json.loads(self._content)
