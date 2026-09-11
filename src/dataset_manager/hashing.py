"""Cryptographic hashing and canonical JSON utilities for Dataset Manager."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """Return hex-encoded SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    """Return hex-encoded SHA-256 digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_dumps(value: Any) -> str:
    """Deterministic JSON string per canonical data model specification."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Deterministic UTF-8 JSON serialization per canonical data model specification."""
    return canonical_json_dumps(value).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    """Return hex-encoded SHA-256 digest of canonically serialized JSON data."""
    return sha256_bytes(canonical_json_bytes(value))
