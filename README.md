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
- **512 MB RAM Preprocessing Limitation**:
  - Hugging Face storage solves artifact **durability**, but does **NOT** resolve the in-memory footprint of full CIFAR-10 static preprocessing.
  - The current in-memory preprocessing pipeline requires ~614 MB RAM to unpack, normalize, and partition all 50,000 CIFAR-10 images at once.
  - Running a full CIFAR-10 build on Render Starter (512 MB RAM) will encounter an Out Of Memory (OOM) error until a streaming/chunked preprocessor is implemented.