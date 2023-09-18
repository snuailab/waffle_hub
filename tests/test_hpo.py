import uuid
from pathlib import Path

import pytest

from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub


@pytest.fixture
def hpo_instance():
    hub_name = f"test_{uuid.uuid1()}"
    hub = Hub.new(
        name=hub_name,
        task="classification",
        model_type="yolov8",
        model_size="n",
        backend="ultralytics",
    )
    return hub


@pytest.mark.parametrize(
    "n_trials, hpo_method, search_space, direction, epochs, batch_size",
    [
        (
            2,
            "RandomSampler",
            {
                "lr0": [0.005, 0.05],
                "lrf": [0.001, 0.005],
                "mosaic": [0.6, 1],
                "cos_lr": (True, False),
                "hsv_h": [0.01, 0.02],
                "hsv_s": [0.01, 0.02],
                "hsv_v": [0.01, 0.02],
                "translate": [0.09, 0.11],
                "scale": [0.45, 0.55],
            },
            "minimize",
            2,
            32,
        ),
        (
            2,
            "RandomSampler",
            {
                "lr0": [0.005, 0.05],
                "lrf": [0.001, 0.005],
                "mosaic": [0.6, 1],
                "cos_lr": (True, False),
                "hsv_h": [0.01, 0.02],
                "hsv_s": [0.01, 0.02],
                "hsv_v": [0.01, 0.02],
                "translate": [0.09, 0.11],
                "scale": [0.45, 0.55],
            },
            "maximize",
            2,
            32,
        ),
        (
            2,
            "GridSampler",
            {
                "lr0": [0.005, 0.05],
                "lrf": [0.001, 0.005],
                "mosaic": [0.6, 1],
                "cos_lr": (True, False),
                "hsv_h": [0.01, 0.02],
            },
            "maximize",
            2,
            32,
        ),
        (
            2,
            "GridSampler",
            {
                "lr0": [0.005, 0.05],
                "lrf": [0.001, 0.005],
                "mosaic": [0.6, 1],
                "cos_lr": (True, False),
                "hsv_h": [0.01, 0.02],
            },
            "minimize",
            2,
            32,
        ),
        (
            5,
            "TPESampler",
            {
                "lr0": [0.005, 0.05],
                "lrf": [0.001, 0.005],
                "mosaic": [0.6, 1],
                "cos_lr": (True, False),
                "hsv_h": [0.01, 0.02],
            },
            "maximize",
            2,
            32,
        ),
        (
            5,
            "BOHB",
            {
                "lr0": [0.005, 0.05],
                "lrf": [0.001, 0.005],
                "mosaic": [0.6, 1],
                "cos_lr": (True, False),
                "hsv_h": [0.01, 0.02],
            },
            "maximize",
            2,
            32,
        ),
    ],
)
def test_hpo(hpo_instance, n_trials, hpo_method, search_space, direction, epochs, batch_size):
    # Act
    dataset = Dataset.load(name="mnist_classification")
    result = hpo_instance.hpo(
        dataset, n_trials, direction, hpo_method, search_space, epochs=epochs, batch_size=batch_size
    )
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
