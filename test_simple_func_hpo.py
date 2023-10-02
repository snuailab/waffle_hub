import time
import uuid
from pathlib import Path

import pytest
from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.core.hpo import OptunaHPO
from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub
from waffle_hub.schema.configs import HPOConfig


def simple_func(x, y, z, a, q, w, e, r, t, u, i, o, **kwargs):
    return x**2 + (y + z + a + q + w + e) / 2 + (r + t + y + u + i + o) * 2


def test_simple_func_hpo():

    search_space = {
        "x": [0.005, 0.05],
        "y": [0.001, 0.005],
        "z": (10, 100, 20),
        "a": (10, 100, 20),
        "q": (10, 100, 20),
        "w": (10, 100, 20),
        "e": (10, 100, 20),
        "r": (10, 100, 20),
        "t": (10, 100, 20),
        "u": (10, 100, 20),
        "i": (10, 100, 20),
        "o": (10, 100, 20),
    }

    name = f"test_{uuid.uuid1()}"
    hpo = OptunaHPO(
        study_name=name,
        root_dir="/home/daeun/workspace/waffle_hub/hubs",
        sampler="randomsampler",
        pruner="nopruner",
        direction="maximize",
        n_trials=10,
        search_space=search_space,
    )
    hpo.run_hpo(objective=simple_func)


if __name__ == "__main__":
    test_simple_func_hpo()
