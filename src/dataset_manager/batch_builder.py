"""Fixed-size physical batch construction and pickle-free NPZ persistence."""

import os
from pathlib import Path

import numpy as np

from dataset_manager.hashing import sha256_file
from dataset_manager.preprocessing import SampleSource


class BatchBuilder:
    def split(
        self, partitions: tuple[np.ndarray, ...], batch_size: int
    ) -> tuple[tuple[np.ndarray, ...], ...]:
        if type(batch_size) is not int or batch_size <= 0 or not partitions:
            raise ValueError("Invalid batch size or empty partitions")
        if any(partition.ndim != 1 for partition in partitions):
            raise ValueError("Physical shard partitions must be one-dimensional")
        shards: list[tuple[np.ndarray, ...]] = []
        for partition in partitions:
            shards.append(
                tuple(
                    partition[offset : offset + batch_size].copy()
                    for offset in range(0, len(partition), batch_size)
                )
            )
        return tuple(shards)

    def write(self, path: Path, samples: SampleSource, indices: np.ndarray) -> dict[str, object]:
        if indices.dtype != np.int64 or indices.ndim != 1 or not len(indices):
            raise ValueError("Invalid sample selection")
        if (
            len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= samples.sample_count)
        ):
            raise ValueError("Invalid or duplicated sample index")
        selected = samples.take(indices)
        x, y, ids = selected.x, selected.y, selected.sample_ids
        if x.dtype != np.float32 or y.dtype != np.int64 or ids.dtype != np.int64:
            raise ValueError("Canonical NPZ arrays must not require pickle")
        with Path(path).open("xb") as stream:
            np.savez(stream, x=x, y=y, sample_ids=ids)
            stream.flush()
            os.fsync(stream.fileno())
        return {
            "sample_count": len(indices),
            "byte_size": Path(path).stat().st_size,
            "sha256": sha256_file(str(path)),
        }
