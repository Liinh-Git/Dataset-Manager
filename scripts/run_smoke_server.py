from pathlib import Path

import numpy as np
import uvicorn

from dataset_manager.app import create_app
from dataset_manager.config import DatasetBuildConfig, DatasetManagerConfig
from dataset_manager.preprocessing import Preprocessor
from dataset_manager.service import BuildExecutionResult, DatasetService
from dataset_manager.storage import DatasetStorage


def fake_executor(store_root: Path):
    def _exec(build_id, request, update):
        config = DatasetBuildConfig(
            1,
            build_id,
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
        raw = np.arange(6 * 12, dtype=np.uint8).reshape(6, 3, 2, 2)
        samples = Preprocessor((3, 2, 2), 10, (0.5,) * 3, (0.5,) * 3).transform(
            raw, np.arange(6, dtype=np.int64)
        )
        published = DatasetStorage(store_root).materialize(config, samples)
        return BuildExecutionResult(published)

    return _exec


def main():
    store_dir = Path("var/smoke_store").resolve()
    temp_dir = Path("var/smoke_temp").resolve()
    store_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    config = DatasetManagerConfig(
        host="127.0.0.1",
        port=9200,
        store_dir=str(store_dir),
        temp_dir=str(temp_dir),
        public_base_url="http://127.0.0.1:9200",
    )
    service = DatasetService(config, executor=fake_executor(store_dir))
    app = create_app(config, service)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
