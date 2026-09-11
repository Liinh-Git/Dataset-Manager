"""Equal-K physical batch construction and pickle-free NPZ persistence."""

import os
from pathlib import Path

import numpy as np

from dataset_manager.hashing import sha256_file
from dataset_manager.preprocessing import Samples


class BatchBuilder:
    def split(
        self, partitions: tuple[np.ndarray, ...], batch_size: int
    ) -> tuple[tuple[np.ndarray, ...], ...]:
        if type(batch_size) is not int or batch_size <= 0 or not partitions:
            raise ValueError("Invalid batch size or empty partitions")
        counts = [len(p) for p in partitions]
        maximum = max(counts)
        k = (maximum + batch_size - 1) // batch_size
        if k == 0 or min(counts) < k:
            raise ValueError("Equal-K would create an empty physical batch")
        shards = []
        for partition in partitions:
            q, r = divmod(len(partition), k)
            offset = 0
            batches = []
            for batch_id in range(k):
                size = q + (batch_id < r)
                if not 0 < size <= batch_size:
                    raise ValueError("Invalid equal-K physical batch")
                batches.append(partition[offset : offset + size].copy())
                offset += size
            shards.append(tuple(batches))
        return tuple(shards)

    def write(self, path: Path, samples: Samples, indices: np.ndarray) -> dict[str, object]:
        if indices.dtype != np.int64 or indices.ndim != 1 or not len(indices):
            raise ValueError("Invalid sample selection")
        if (
            len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(samples.x))
        ):
            raise ValueError("Invalid or duplicated sample index")
        x, y, ids = samples.x[indices], samples.y[indices], samples.sample_ids[indices]
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
