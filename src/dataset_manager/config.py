"""Dataset Manager service and immutable V1 build configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class DatasetManagerConfig:
    host: str = "127.0.0.1"
    port: int = 9200
    store_dir: str = "var/datasets"
    temp_dir: str = "var/datasets-tmp"
    max_source_bytes: int = 512 * 1024 * 1024
    queue_capacity: int = 8
    download_chunk_size: int = 1024 * 1024
    download_parallelism: int = 8
    public_base_url: str = "http://127.0.0.1:9200"
    source_download_timeout_seconds: float = 30.0
    log_level: str = "INFO"
    storage_backend: str = "local"
    hf_repo_id: str | None = None
    hf_token: str | None = None
    hf_branch: str = "main"

    def __post_init__(self) -> None:
        if (
            not self.host
            or not 0 < self.port < 65536
            or self.max_source_bytes <= 0
            or self.queue_capacity <= 0
            or self.download_chunk_size <= 0
            or self.download_parallelism <= 0
            or not self.public_base_url.startswith(("http://", "https://"))
            or self.source_download_timeout_seconds <= 0
        ):
            raise ValueError("Invalid Dataset Manager configuration")
        if Path(self.store_dir).resolve() == Path(self.temp_dir).resolve():
            raise ValueError("Published and temporary roots must differ")
        if self.storage_backend not in ("local", "huggingface"):
            raise ValueError(f"Invalid storage backend: {self.storage_backend}")
        if self.storage_backend == "huggingface":
            if (
                not self.hf_repo_id
                or not isinstance(self.hf_repo_id, str)
                or not self.hf_repo_id.strip()
            ):
                raise ValueError("HF_REPO_ID is required when storage_backend is 'huggingface'")
            if not self.hf_token or not isinstance(self.hf_token, str) or not self.hf_token.strip():
                raise ValueError("HF_TOKEN is required when storage_backend is 'huggingface'")
            if (
                not self.hf_branch
                or not isinstance(self.hf_branch, str)
                or not self.hf_branch.strip()
            ):
                raise ValueError("HF_BRANCH must be a non-empty branch name")
            stripped = self.hf_branch.strip()
            if len(stripped) == 40 and all(c in "0123456789abcdefABCDEF" for c in stripped):
                raise ValueError(
                    "HF_BRANCH must denote a writable branch, not an immutable commit SHA"
                )

    def __repr__(self) -> str:
        token_repr = "'***'" if self.hf_token else "None"
        return (
            f"DatasetManagerConfig(host={self.host!r}, port={self.port}, "
            f"store_dir={self.store_dir!r}, temp_dir={self.temp_dir!r}, "
            f"storage_backend={self.storage_backend!r}, hf_repo_id={self.hf_repo_id!r}, "
            f"hf_token={token_repr}, hf_branch={self.hf_branch!r}, "
            f"public_base_url={self.public_base_url!r})"
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> "DatasetManagerConfig":
        """Construct configuration with environment variable defaults."""
        host = overrides.get("host", os.environ.get("HOST", "127.0.0.1"))
        raw_port = overrides.get("port", os.environ.get("PORT", "9200"))
        port = int(raw_port)
        store_dir = overrides.get("store_dir", os.environ.get("STORE_DIR", "var/datasets"))
        temp_dir = overrides.get("temp_dir", os.environ.get("TEMP_DIR", "var/datasets-tmp"))
        public_base_url = overrides.get(
            "public_base_url",
            os.environ.get("PUBLIC_BASE_URL", f"http://{host}:{port}"),
        )
        log_level = overrides.get("log_level", os.environ.get("LOG_LEVEL", "INFO"))
        storage_backend = overrides.get(
            "storage_backend", os.environ.get("DATASET_STORAGE_BACKEND", "local")
        )
        hf_repo_id = overrides.get("hf_repo_id", os.environ.get("HF_REPO_ID"))
        hf_token = overrides.get("hf_token", os.environ.get("HF_TOKEN"))
        hf_branch = overrides.get("hf_branch", os.environ.get("HF_BRANCH", "main"))
        return cls(
            host=host,
            port=port,
            store_dir=store_dir,
            temp_dir=temp_dir,
            public_base_url=public_base_url,
            log_level=log_level,
            storage_backend=storage_backend,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token,
            hf_branch=hf_branch,
        )


@dataclass(frozen=True, slots=True)
class DatasetBuildConfig:
    schema_version: int
    dataset_build_id: str
    dataset_id: str
    name: str
    profile: str
    task_type: str
    input_shape: tuple[int, int, int]
    dtype: str
    num_classes: int
    preprocessing: dict[str, object]
    batch_size: int
    shard_count: int
    partition_algorithm: str
    partition_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(self, "preprocessing", _freeze_json(self.preprocessing))
        if (
            type(self.schema_version) is not int
            or self.schema_version <= 0
            or not self.dataset_build_id
            or not self.dataset_id
            or not self.name
            or not self.profile
            or not self.task_type
            or len(self.input_shape) != 3
            or any(v <= 0 for v in self.input_shape)
            or self.dtype != "float32"
            or self.num_classes <= 0
            or self.batch_size <= 0
            or self.shard_count <= 0
            or self.partition_algorithm != "seeded_permutation_round_robin"
            or type(self.partition_seed) is not int
        ):
            raise ValueError("Invalid pinned Dataset Build configuration")
