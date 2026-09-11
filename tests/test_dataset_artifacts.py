"""Deterministic Dataset Build artifact tests, independent of Backend and Runtime."""

from dataclasses import replace

import numpy as np
import pytest

from dataset_manager.batch_builder import BatchBuilder
from dataset_manager.config import DatasetBuildConfig
from dataset_manager.importer import DatasetImporter
from dataset_manager.manifest import DatasetManifest, ManifestBuilder
from dataset_manager.partitioner import Partitioner
from dataset_manager.preprocessing import Preprocessor
from dataset_manager.storage import DatasetStorage


def config(**changes):
    base = DatasetBuildConfig(
        1,
        "build",
        "cifar10",
        "CIFAR-10",
        "CNN_IMAGE_CLASSIFICATION_V1",
        "image_classification",
        (3, 2, 2),
        "float32",
        10,
        {
            "channel_order": "NCHW",
            "scale": "uint8_to_unit",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        2,
        3,
        "seeded_permutation_round_robin",
        42,
    )
    return replace(base, **changes)


def samples(n=10):
    x = np.arange(n * 12, dtype=np.uint8).reshape(n, 3, 2, 2)
    y = np.arange(n, dtype=np.int64) % 10
    return Preprocessor((3, 2, 2), 10, (0.5,) * 3, (0.5,) * 3).transform(x, y)


def test_binary_cifar_import_and_source_validation(tmp_path):
    labels = np.array([0, 9], dtype=np.uint8)
    images = np.arange(2 * 3072, dtype=np.uint16).astype(np.uint8).reshape(2, 3072)
    records = np.concatenate((labels[:, None], images), axis=1)
    source = tmp_path / "data_batch.bin"
    source.write_bytes(records.tobytes())
    imported = DatasetImporter().import_cifar10_binary((source,))
    assert imported.x.shape == (2, 3, 32, 32)
    assert imported.x.dtype == np.uint8
    np.testing.assert_array_equal(imported.y, [0, 9])
    np.testing.assert_array_equal(imported.sample_ids, [0, 1])
    source.write_bytes(b"broken")
    with pytest.raises(ValueError):
        DatasetImporter().import_cifar10_binary((source,))
    source.write_bytes(bytes([10]) + bytes(3072))
    with pytest.raises(ValueError):
        DatasetImporter().import_cifar10_binary((source,))
    with pytest.raises(ValueError):
        DatasetImporter().import_cifar10_binary(())


def test_preprocess_rejects_empty_shape_and_labels():
    normalized = samples(2)
    assert normalized.x.dtype == np.float32
    assert normalized.y.dtype == normalized.sample_ids.dtype == np.int64
    np.testing.assert_array_equal(normalized.sample_ids, [0, 1])
    with pytest.raises(ValueError):
        Preprocessor((3, 2, 2), 10, (0.5,) * 3, (0.5,) * 3).transform(
            np.empty((0, 3, 2, 2), dtype=np.uint8), np.empty(0, dtype=np.int64)
        )
    with pytest.raises(ValueError):
        Preprocessor((3, 2, 2), 10, (0.5,) * 3, (0.5,) * 3).transform(
            np.zeros((1, 3, 2, 2), dtype=np.uint8), np.array([10])
        )
    with pytest.raises(ValueError):
        Preprocessor((3, 2, 2), 10, (0.5,) * 3, (0.5,) * 3).transform(
            np.zeros((1, 3, 3, 3), dtype=np.uint8), np.array([1])
        )


@pytest.mark.parametrize("n", [3, 4, 10, 17])
def test_partition_is_deterministic_disjoint_and_complete(n):
    first = Partitioner().partition(n, 3, 42)
    second = Partitioner().partition(n, 3, 42)
    different = Partitioner().partition(n, 3, 43)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    if n > 3:
        assert any(not np.array_equal(a, b) for a, b in zip(first, different, strict=True))
    flat = np.concatenate(first)
    assert sorted(flat) == list(range(n))
    assert len(np.unique(flat)) == n
    assert max(map(len, first)) - min(map(len, first)) <= 1


def test_equal_k_quotient_remainder_has_no_padding():
    partitions = (
        np.arange(0, 5, dtype=np.int64),
        np.arange(5, 9, dtype=np.int64),
        np.arange(9, 13, dtype=np.int64),
    )
    batches = BatchBuilder().split(partitions, 2)
    assert [list(map(len, shard)) for shard in batches] == [[2, 2, 1], [2, 1, 1], [2, 1, 1]]
    assert all(len(shard) == 3 for shard in batches)
    assert sorted(np.concatenate([batch for shard in batches for batch in shard])) == list(
        range(13)
    )
    for invalid in (0, -1, True):
        with pytest.raises(ValueError):
            BatchBuilder().split(partitions, invalid)
    with pytest.raises(ValueError):
        BatchBuilder().split((np.array([0, 1]), np.array([2])), 1)


def test_build_tree_is_deterministic_verified_and_registering(tmp_path):
    frozen = config()
    with pytest.raises(TypeError):
        frozen.preprocessing["mean"] = [0, 0, 0]
    with pytest.raises(TypeError):
        frozen.preprocessing["mean"][0] = 0
    first = DatasetStorage(tmp_path / "a").materialize(frozen, samples())
    second = DatasetStorage(tmp_path / "b").materialize(config(), samples())
    assert first.dataset_manifest_hash == second.dataset_manifest_hash
    assert first.lifecycle_state == "REGISTERING"
    manifest = DatasetStorage(tmp_path / "a").verify(first)
    root = manifest.value
    assert root["sample_count"] == 10
    assert root["batch_count_per_shard"] == 2
    assert [s["sample_count"] for s in root["shards"]] == [4, 3, 3]
    assert all(s["batch_count"] == 2 for s in root["shards"])
    again = DatasetStorage(tmp_path / "a").materialize(config(), samples())
    assert again == first
    assert not list((tmp_path / "a" / ".tmp").iterdir())


@pytest.mark.parametrize("target", ["batch", "shard", "root"])
def test_corruption_breaks_hash_chain(tmp_path, target):
    storage = DatasetStorage(tmp_path)
    published = storage.materialize(config(), samples())
    root = DatasetManifest(published.manifest_path.read_bytes()).value
    if target == "root":
        path = published.manifest_path
    else:
        shard_path = published.directory / root["shards"][0]["relative_shard_manifest_path"]
        if target == "shard":
            path = shard_path
        else:
            shard = DatasetManifest(shard_path.read_bytes()).value
            path = published.directory / shard["batches"][0]["relative_filename"]
    content = path.read_bytes()
    path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises((ValueError, OSError)):
        storage.verify(published)


def test_path_traversal_symlink_and_immutable_conflict(tmp_path):
    storage = DatasetStorage(tmp_path)
    published = storage.materialize(config(), samples())
    for value in ("../secret", "/absolute", r"shards\..\secret", "."):
        with pytest.raises(ValueError):
            storage.resolve_artifact(published, value)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    link = published.directory / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pass
    else:
        with pytest.raises(ValueError):
            storage.resolve_artifact(published, "escape/file")
    with pytest.raises(ValueError):
        storage.materialize(replace(config(), partition_seed=999), samples())


def test_write_failure_never_publishes_partial_build(tmp_path, monkeypatch):
    storage = DatasetStorage(tmp_path)
    original = BatchBuilder.write
    calls = 0

    def fail_second(self, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        return original(self, *args)

    monkeypatch.setattr(BatchBuilder, "write", fail_second)
    with pytest.raises(OSError):
        storage.materialize(config(), samples())
    with pytest.raises(ValueError):
        storage.load("build")
    assert not list((tmp_path / ".tmp").iterdir())


def test_manifest_rejects_unsafe_paths_and_noncanonical_json():
    builder = ManifestBuilder()
    with pytest.raises(ValueError):
        builder.shard(
            "build",
            0,
            (
                {
                    "batch_id": 0,
                    "relative_filename": "../batch.npz",
                    "sample_count": 1,
                    "byte_size": 1,
                    "sha256": "a" * 64,
                },
            ),
        )
    with pytest.raises(ValueError):
        DatasetManifest(b'{"z": 1, "a": 2}')


# ── Issue E: DatasetStorage path validation tests ─────────────────────────────


def test_e_storage_safe_rejects_invalid_paths(tmp_path):
    """E: _safe() rejects absolute paths, '..', NUL, backslash, and empty paths."""
    storage = DatasetStorage(tmp_path)
    published = storage.materialize(config(), samples())

    # Absolute path
    with pytest.raises(ValueError, match="absolute"):
        storage.resolve_artifact(published, "/absolute/path")

    # Parent-directory traversal
    with pytest.raises(ValueError):
        storage.resolve_artifact(published, "../outside")

    with pytest.raises(ValueError):
        storage.resolve_artifact(published, "shards/../../outside")

    # Backslash (cross-platform traversal ambiguity)
    with pytest.raises(ValueError, match="forward-slash"):
        storage.resolve_artifact(published, "shards\\..\\outside")

    # NUL byte
    with pytest.raises(ValueError, match="NUL"):
        storage.resolve_artifact(published, "shards/\x00batch")

    # Windows drive letter (lexical check)
    with pytest.raises(ValueError, match="absolute"):
        storage.resolve_artifact(published, "C:/something")

    # Empty string
    with pytest.raises(ValueError):
        storage.resolve_artifact(published, "")


def test_e_storage_safe_accepts_canonical_paths(tmp_path):
    """E: _safe() accepts legitimate canonical artifact relative paths."""
    storage = DatasetStorage(tmp_path)
    published = storage.materialize(config(), samples())
    root = DatasetStorage(tmp_path).verify(published).value
    shard_ref = root["shards"][0]["relative_shard_manifest_path"]
    # Should not raise
    path = storage.resolve_artifact(published, shard_ref)
    assert path.is_file()


# ── Issue F: Existing build full verification before reuse ────────────────────


def test_f_corrupt_existing_build_detected_before_reuse(tmp_path):
    """F: materialize() fast-path detects corruption before returning existing build."""
    storage = DatasetStorage(tmp_path)
    published = storage.materialize(config(), samples())

    # Corrupt a batch artifact while leaving root manifest untouched
    root = DatasetStorage(tmp_path).verify(published).value
    shard_path = published.directory / root["shards"][0]["relative_shard_manifest_path"]
    shard = __import__("json").loads(shard_path.read_bytes())
    batch_path = published.directory / shard["batches"][0]["relative_filename"]
    original = batch_path.read_bytes()
    batch_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))

    # Call materialize again with same build identity/config
    with pytest.raises(ValueError):
        storage.materialize(config(), samples())
