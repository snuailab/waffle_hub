import time
import uuid
from pathlib import Path

import pytest
from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.core.hpo import OptunaHPO
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub


def test_classification_hpo():
    hold = True
    dataset = Dataset.load(
        name="mnist_classification", root_dir="/home/daeun/workspace/waffle_hub/datasets"
    )
    search_space = {
        "lr0": [0.005, 0.05],
        "lrf": [0.001, 0.005],
        "mosaic": [0.6, 1],
        "cos_lr": (True, False),
        "hsv_h": [0.01, 0.02],
        "epochs": (3, 2, 1),
    }

    n_trials = 2
    direction = "maximize"
    sampler = {"randomsampler": {"seed": 42}}
    pruner = {"nopruner": {"min_resource": 1, "reduction_factor": 4}}

    name = f"test_{uuid.uuid1()}"
    hub = Hub.new(
        name=name,
        backend="ultralytics",
        task=TaskType.CLASSIFICATION,
        model_type="yolov8",
        model_size="n",
        categories=dataset.get_category_names(),
        device="cpu",
        workers=0,
        hold=hold,
    )

    result = hub.hpo(
        dataset="mnist_classification",
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        n_trials=n_trials,
        metric="accuracy",
        search_space=search_space,
        image_size=24,
        batch_size=24,
    )


if __name__ == "__main__":
    test_classification_hpo()
