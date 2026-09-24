# Dataset Manager

Standalone Dataset Ingestion, Partitioning, and Serving Service.

## Overview
Dataset Manager builds, verifies, and serves partition shards and batch NPZ artifacts for distributed training workers via standard HTTP endpoints (`/api/v1/...`, `/artifacts/v1/...`).

## Local Execution
```bash
# Install dependencies
uv sync

# Run service locally with local storage
uv run dataset-manager --host 127.0.0.1 --port 9200

# Run service with Hugging Face durable storage backend
export DATASET_STORAGE_BACKEND=huggingface
export HF_REPO_ID=your-username/dataset-artifacts
export HF_TOKEN=hf_yourWriteToken
export HF_BRANCH=main
uv run dataset-manager --host 127.0.0.1 --port 9200
```

V1 assumes Dataset Manager itself is local/private. It does not implement an
application authentication layer.

## Storage Backends
Dataset Manager supports two durable storage backends:
1. **Local Filesystem (`local`, default)**:
   - Persists all manifests and NPZ batch artifacts in `STORE_DIR`.
   - Requires persistent volume storage to survive container restarts.
2. **Hugging Face Dataset Repository (`huggingface`)**:
   - Uses a remote Hugging Face dataset repository as the authoritative durable storage.
   - Preserves authoritative layout under `dataset-builds/{dataset_build_id}/` and metadata under `.service/builds/{dataset_build_id}.json`.
   - Local directory is used as an ephemeral staging workspace and read-through verified cache.
   - On cache miss, manifests and NPZ batches are fetched on-demand from the remote repo with cryptographic SHA-256 validation against the root manifest hash chain.
   - `HF_TOKEN` must be provided via environment variable only (never via CLI flags).
   - `HF_TOKEN` is used only by Dataset Manager to publish/write. Workers do not
     need it when the repository is publicly readable.
   - For this backend, Dataset Manager publishes the Hugging Face `resolve` URL
     as `artifact_base_url`; that provider-specific construction stays inside
     `HuggingFaceArtifactStore`. Manifest-relative shard and batch paths identify
     the files beneath that origin.

The domain requirement is a durable `ArtifactStore`; Hugging Face is one current
implementation and deployment choice. Current PBL4 `dev` does not yet implement
direct-HF manifest-relative provisioning, all-shards caching, or `work_units[]`.

`PUBLIC_BASE_URL` remains the local-storage fallback: when artifacts are served
by Dataset Manager itself, it forms the advertised `/artifacts/v1/...` URLs. It
is not a WAN or production-mode requirement for the Hugging Face backend.

## Running Tests
```bash
# Run all unit and integration tests
uv run pytest tests/

# Run Hugging Face real remote smoke test (skips cleanly if HF_TOKEN is absent)
uv run python scripts/real_hf_smoke.py
```

## Deployment Notes (Render)
- **Start Command**: `dataset-manager --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/healthz`
- **Container Durability**:
  - With `DATASET_STORAGE_BACKEND=huggingface`, published builds survive Render container restarts without requiring a paid Persistent Disk ($0.25/GB/mo).
- **Bounded-memory preprocessing**:
  - CIFAR-10 source files are validated and memory-mapped.
  - Static normalization materializes one physical batch at a time, avoiding a
    full-dataset `float32` duplicate while preserving deterministic artifacts.
- **Physical batches**:
  - Each shard is sliced into configured-size batches plus at most one smaller
    tail, with no padding, duplication, or dropped samples.
  - `CNN_IMAGE_CLASSIFICATION_V1` currently requires three shards and equal
    batch counts; incompatible configurations are rejected after generic
    batching rather than having their batch sizes smeared.
