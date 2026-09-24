"""Profile-specific Dataset Build constraints kept outside generic pipeline logic."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    name: str
    task_type: str
    input_shape: tuple[int, int, int]
    num_classes: int
    shard_count: int
    requires_equal_batch_count_per_shard: bool

    def validate_request(self, input_shape: tuple[int, int, int], shard_count: int) -> None:
        if input_shape != self.input_shape or shard_count != self.shard_count:
            raise ValueError(
                f"{self.name} requires input_shape={list(self.input_shape)} "
                f"and shard_count={self.shard_count}"
            )

    def validate_batch_counts(self, batch_counts: tuple[int, ...]) -> None:
        if not batch_counts:
            raise ValueError("A Dataset Build must contain physical shards")
        if self.requires_equal_batch_count_per_shard and len(set(batch_counts)) != 1:
            raise ValueError(
                f"{self.name} requires equal batch_count_per_shard; "
                f"fixed-size materialization produced {batch_counts}"
            )


CNN_IMAGE_CLASSIFICATION_V1 = DatasetProfile(
    name="CNN_IMAGE_CLASSIFICATION_V1",
    task_type="image_classification",
    input_shape=(3, 32, 32),
    num_classes=10,
    shard_count=3,
    requires_equal_batch_count_per_shard=True,
)


def resolve_profile(name: str) -> DatasetProfile:
    if name == CNN_IMAGE_CLASSIFICATION_V1.name:
        return CNN_IMAGE_CLASSIFICATION_V1
    raise ValueError(f"Unsupported Dataset Build profile: {name}")


def validate_materialized_batch_counts(name: str, batch_counts: tuple[int, ...]) -> None:
    """Apply known profile layout rules after generic fixed-size batching."""
    if name == CNN_IMAGE_CLASSIFICATION_V1.name:
        CNN_IMAGE_CLASSIFICATION_V1.validate_batch_counts(batch_counts)
