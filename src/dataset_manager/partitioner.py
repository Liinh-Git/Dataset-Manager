"""Seeded sample permutation followed by round-robin shard assignment."""

from random import Random

import numpy as np


class Partitioner:
    def partition(
        self, sample_count: int, shard_count: int, partition_seed: int
    ) -> tuple[np.ndarray, ...]:
        if (
            type(sample_count) is not int
            or sample_count <= 0
            or type(shard_count) is not int
            or shard_count <= 0
            or type(partition_seed) is not int
        ):
            raise ValueError("Invalid partition configuration")
        order = list(range(sample_count))
        # A local PRNG cannot alter unrelated caller/global random state.
        Random(partition_seed).shuffle(order)
        return tuple(np.asarray(order[i::shard_count], dtype=np.int64) for i in range(shard_count))
