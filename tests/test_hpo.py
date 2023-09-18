from pathlib import Path

import pytest

from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub


def test_hpo():
    n_trials = 2
    hpo_method = "RandomSampler"
    search_space = {
        "lr0": [0.005, 0.05],
        "lrf": [0.001, 0.005],
        "mosaic": [0.6, 1],
        "cos_lr": (True, False),
        "hsv_h": [0.01, 0.02],
        "hsv_s": [0.01, 0.02],
        "hsv_v": [0.01, 0.02],
        "translate": [0.09, 0.11],
        "scale": [0.45, 0.55],
    }
    hub_name = "test6"

    hub = Hub.new(
        name=hub_name,
        task="classification",
        model_type="yolov8",
        model_size="n",
        backend="ultralytics",
    )

    dataset = Dataset.load(name="mnist_classification")
    direction = "minimize"

    result = hub.hpo(
        dataset=dataset,
        n_trials=n_trials,
        direction=direction,
        hpo_method=hpo_method,
        search_space=search_space,
        epochs=30,
        image_size=24,
        device="0",
    )
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result


if __name__ == "__main__":
    test_hpo()
