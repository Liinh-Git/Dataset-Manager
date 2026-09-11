"""Static, deterministic preparation of canonical image samples."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Samples:
    x: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray


class Preprocessor:
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ):
        self.input_shape = tuple(input_shape)
        self.num_classes = num_classes
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if (
            len(self.input_shape) != 3
            or any(type(v) is not int or v <= 0 for v in self.input_shape)
            or type(num_classes) is not int
            or num_classes <= 0
            or self.mean.shape != (self.input_shape[0],)
            or self.std.shape != self.mean.shape
            or np.any(self.std <= 0)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
        ):
            raise ValueError("Invalid static preprocessing configuration")

    def transform(self, images: np.ndarray, labels: np.ndarray) -> Samples:
        """CIFAR canonical source is uint8 NCHW; ordinals follow source order."""
        images = np.asarray(images)
        labels = np.asarray(labels)
        if images.dtype != np.uint8 or images.ndim != 4 or not len(images):
            raise ValueError("Expected nonempty uint8 NCHW input")
        if tuple(images.shape[1:]) != self.input_shape:
            raise ValueError("Source shape does not match the pinned preprocessing profile")
        if (
            labels.dtype.kind not in "iu"
            or labels.shape != (len(images),)
            or np.any(labels < 0)
            or np.any(labels >= self.num_classes)
        ):
            raise ValueError("Invalid class labels")
        values = images.astype(np.float32) / np.float32(255)
        values = (values - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite preprocessed values")
        return Samples(values, labels.astype(np.int64), np.arange(len(images), dtype=np.int64))
