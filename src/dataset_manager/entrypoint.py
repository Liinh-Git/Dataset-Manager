"""Console entrypoint for the standalone Dataset Manager HTTP process."""

import argparse
import os

import uvicorn

from dataset_manager.app import create_app
from dataset_manager.config import DatasetManagerConfig


def main() -> None:
    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", "9200"))
    default_store = os.environ.get("STORE_DIR", "var/datasets")
    default_temp = os.environ.get("TEMP_DIR", "var/datasets-tmp")
    default_public_url = os.environ.get("PUBLIC_BASE_URL")

    parser = argparse.ArgumentParser(
        prog="dataset-manager",
        description="Dataset Ingestion, Partitioning, and Serving Service.",
    )
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--store-dir", default=default_store)
    parser.add_argument("--temp-dir", default=default_temp)
    parser.add_argument("--queue-capacity", type=int, default=8)
    parser.add_argument("--public-base-url", default=default_public_url)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    public_url = args.public_base_url or f"http://{args.host}:{args.port}"
    config = DatasetManagerConfig(
        host=args.host,
        port=args.port,
        store_dir=args.store_dir,
        temp_dir=args.temp_dir,
        queue_capacity=args.queue_capacity,
        public_base_url=public_url,
        log_level=args.log_level.upper(),
    )
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
