import uuid

from waffle_hub.core.hpo.adapter.optuna import OptunaHPO


def simple_func(x, y, z, a, q, w, e, r, t, u, i, o, **kwargs):
    print(x, y)
    return x**2 + (y + z + a + q + w + e) / 2 + (r + t + y + u + i + o) * 2


def test_simple_func_hpo():

    search_space_config = {
        "x": {
            "method": "suggest_categorical",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "y": {
            "method": "suggest_categorical",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "z": {
            "method": "suggest_categorical",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "a": {
            "method": "suggest_categorical",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "q": {
            "method": "suggest_float",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "w": {
            "method": "suggest_float",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "e": {
            "method": "suggest_float",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "e": {
            "method": "suggest_float",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "r": {
            "method": "suggest_float",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "t": {
            "method": "suggest_float",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "u": {
            "method": "suggest_float",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "i": {
            "method": "suggest_float",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
        "o": {
            "method": "suggest_float",
            "search_space": [0.005, 0.05],
            "kwargs": {},
        },
    }

    sampler = {"TPESampler": {"n_startup_trials": 20, "multivariate": False}}
    pruner = {"MedianPruner": {}}

    name = f"test_{uuid.uuid1()}"
    hpo = OptunaHPO(
        study_name=name,
        root_dir="/home/daeun/workspace/waffle_hub/hubs",
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        n_trials=100,
        search_space=search_space_config,
    )
    hpo.run_hpo(objective=simple_func)


if __name__ == "__main__":
    test_simple_func_hpo()
