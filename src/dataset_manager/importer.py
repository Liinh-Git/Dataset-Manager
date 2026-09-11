"""CIFAR-10 binary ingestion without torchvision or pickle deserialization."""

from pathlib import Path

import numpy as np

from dataset_manager.preprocessing import Samples

_RECORD_BYTES = 1 + 3 * 32 * 32


class DatasetImporter:
    """Decode the official CIFAR-10 binary record representation."""

    def import_cifar10_binary(self, files: tuple[Path, ...]) -> Samples:
        paths = tuple(Path(path) for path in files)
        if not paths:
            raise ValueError("At least one CIFAR-10 binary batch is required")
        images: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError("CIFAR-10 source must be a regular file")
            raw = path.read_bytes()
            if not raw or len(raw) % _RECORD_BYTES:
                raise ValueError("Corrupt CIFAR-10 binary batch")
            records = np.frombuffer(raw, dtype=np.uint8).reshape(-1, _RECORD_BYTES)
            batch_labels = records[:, 0].astype(np.int64)
            if np.any(batch_labels >= 10):
                raise ValueError("CIFAR-10 label is outside [0, 9]")
            labels.append(batch_labels)
            # Official bytes store 1024 red, then green, then blue values.
            images.append(records[:, 1:].reshape(-1, 3, 32, 32).copy())
        x = np.concatenate(images)
        y = np.concatenate(labels)
        return Samples(x, y, np.arange(len(x), dtype=np.int64))
