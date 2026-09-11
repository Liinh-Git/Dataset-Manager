# Dataset Manager

Standalone Dataset Ingestion, Partitioning, and Serving Service.

## Overview
Dataset Manager builds, verifies, and serves partition shards and batch NPZ artifacts for distributed training workers via standard HTTP endpoints.

## Local Execution
```bash
# Install dependencies
uv sync

# Run service locally
uv run dataset-manager --host 127.0.0.1 --port 9200
```

## Running Tests
```bash
uv run pytest tests/
```

## Deployment Notes (Render)
- **Start Command**: `dataset-manager --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/healthz`
- **Storage**: Requires a persistent disk mounted at `STORE_DIR` (e.g. `/var/data/datasets`) for restart-safe durability of manifests and published artifacts.
- **Memory Requirement**: Full CIFAR-10 float32 preprocessing requires >614 MB RAM. Running full dataset builds on a 512 MB RAM instance will trigger an Out Of Memory (OOM) error. For production CIFAR builds, use a machine tier with at least 1–2 GB RAM or streaming preprocessing.
