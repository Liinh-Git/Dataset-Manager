"""CIFAR-10 binary ingestion without torchvision or pickle deserialization."""

from pathlib import Path

import numpy as np

from dataset_manager.preprocessing import Samples

_RECORD_BYTES = 1 + 3 * 32 * 32


class Cifar10BinarySampleSource:
    """Validated random-access CIFAR-10 records without a full-dataset copy."""

    def __init__(self, files: tuple[Path, ...]):
        self._files = tuple(Path(path) for path in files)
        if not self._files:
            raise ValueError("At least one CIFAR-10 binary batch is required")
        counts: list[int] = []
        for path in self._files:
            if path.is_symlink() or not path.is_file():
                raise ValueError("CIFAR-10 source must be a regular file")
            size = path.stat().st_size
            if not size or size % _RECORD_BYTES:
                raise ValueError("Corrupt CIFAR-10 binary batch")
            records = np.memmap(path, dtype=np.uint8, mode="r").reshape(-1, _RECORD_BYTES)
            if np.any(records[:, 0] >= 10):
                raise ValueError("CIFAR-10 label is outside [0, 9]")
            counts.append(len(records))
        self._counts = tuple(counts)
        self._offsets = np.cumsum((0, *counts), dtype=np.int64)

    @property
    def sample_count(self) -> int:
        return int(self._offsets[-1])

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (3, 32, 32)

    def take(self, indices: np.ndarray) -> Samples:
        indices = np.asarray(indices)
        if (
            indices.dtype != np.int64
            or indices.ndim != 1
            or not len(indices)
            or np.any(indices < 0)
            or np.any(indices >= self.sample_count)
        ):
            raise ValueError("Invalid CIFAR-10 sample selection")
        images = np.empty((len(indices), 3, 32, 32), dtype=np.uint8)
        labels = np.empty(len(indices), dtype=np.int64)
        for file_index, path in enumerate(self._files):
            start = int(self._offsets[file_index])
            stop = int(self._offsets[file_index + 1])
            positions = np.flatnonzero((indices >= start) & (indices < stop))
            if not len(positions):
                continue
            records = np.memmap(path, dtype=np.uint8, mode="r").reshape(-1, _RECORD_BYTES)
            selected = records[indices[positions] - start]
            labels[positions] = selected[:, 0]
            images[positions] = selected[:, 1:].reshape(-1, 3, 32, 32)
        return Samples(images, labels, indices.copy())


class DatasetImporter:
    """Decode the official CIFAR-10 binary record representation."""

    def import_cifar10_binary(self, files: tuple[Path, ...]) -> Samples:
        source = self.open_cifar10_binary(files)
        return source.take(np.arange(source.sample_count, dtype=np.int64))

    def open_cifar10_binary(self, files: tuple[Path, ...]) -> Cifar10BinarySampleSource:
        return Cifar10BinarySampleSource(files)
