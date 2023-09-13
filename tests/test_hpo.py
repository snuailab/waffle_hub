import pytest

from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub


@pytest.fixture
def example_ultralytics_hub():
    hub_name = "test"
    hub = Hub.new(
        name=hub_name,
        task="classification",
        model_type="yolov8",
        model_size="n",
        backend="ultralytics",
    )
    return hub


def test_ultralytics_hpo(example_ultralytics_hub):
    n_trials = 2
    sampler_type = "TPESampler"
    search_space = {
        "lr0": [0.005, 0.05],
        "lrf": [0.001, 0.005],
        "mosaic": [0.6, 1],
        "cos_lr": [True, False],
        "hsv_h": [0.01, 0.02],
        "hsv_s": [0.01, 0.02],
        "hsv_v": [0.01, 0.02],
        "translate": [0.09, 0.11],
        "scale": [0.45, 0.55],
        "mosaic": [0.6, 1],
    }
    dataset = Dataset.load(name="mnist_classification")
    direction = "maximize"

    result = example_ultralytics_hub.hpo(
        dataset=dataset,
        n_trials=n_trials,
        direction=direction,
        sampler_type=sampler_type,
        search_space=search_space,
        epochs=30,
        image_size=8,
        device="0",
    )
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
