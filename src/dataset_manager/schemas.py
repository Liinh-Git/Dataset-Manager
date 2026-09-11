"""Canonical Dataset Manager HTTP request and response models."""

import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetBuildState(StrEnum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    IMPORTING = "IMPORTING"
    VALIDATING = "VALIDATING"
    PREPROCESSING = "PREPROCESSING"
    MATERIALIZING = "MATERIALIZING"
    VERIFYING = "VERIFYING"
    REGISTERING = "REGISTERING"
    READY = "READY"
    FAILED = "FAILED"
    DEPRECATED = "DEPRECATED"
    DELETING = "DELETING"
    DELETED = "DELETED"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cifar10Source(StrictModel):
    type: Literal["cifar10_download"]
    dataset_name: Literal["cifar10"]
    version: Literal["binary-v1"] = "binary-v1"


class Normalization(StrictModel):
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @model_validator(mode="after")
    def validate_values(self):
        if any(not math.isfinite(value) for value in (*self.mean, *self.std)) or any(
            value <= 0 for value in self.std
        ):
            raise ValueError("Normalization values must be finite and std must be positive")
        return self


class CreateBuildRequest(StrictModel):
    source: Cifar10Source
    profile: Literal["CNN_IMAGE_CLASSIFICATION_V1"]
    input_shape: tuple[int, int, int]
    normalization: Normalization
    batch_size: int = Field(gt=0)
    shard_count: int = Field(gt=0)
    partition_seed: int
    command_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_profile(self):
        if self.input_shape != (3, 32, 32) or self.shard_count != 3:
            raise ValueError("CNN_IMAGE_CLASSIFICATION_V1 requires [3,32,32] and 3 shards")
        return self


class RebuildRequest(StrictModel):
    command_id: str = Field(min_length=1)
    batch_size: int | None = Field(default=None, gt=0)
    partition_seed: int | None = None
    normalization: Normalization | None = None
    input_shape: tuple[int, int, int] | None = None


class RegistrationAckRequest(StrictModel):
    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    registration_id: str = Field(min_length=1)
    catalog_persisted_at: str = Field(min_length=1)


class DeprecateRequest(StrictModel):
    reason: str | None = None


class PurgeRequest(StrictModel):
    command_id: str = Field(min_length=1)
    reason: str | None = None
    force: Literal[False] = False


class ApiError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False


class ApiEnvelope(BaseModel):
    request_id: str
    data: Any | None = None
    error: ApiError | None = None
