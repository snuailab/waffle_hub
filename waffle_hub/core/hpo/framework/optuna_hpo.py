import time
from pathlib import Path
from typing import Callable

import optuna
import optuna.visualization as oplt
import plotly.io as pio
from optuna.pruners import HyperbandPruner
from optuna.samplers import GridSampler, RandomSampler, TPESampler

from waffle_hub.core.hpo.base_hpo import BaseHPO
from waffle_hub.schema.configs import HPOMethodConfig


class ObjectiveDirectionMapper:
    def __init__(self, objectives: str):
        self.objectives = objectives
        self._direction = None
        self._objective = None

    @property
    def direction(self):
        return self._direction

    @property
    def objective(self):
        return self._objective

    def set_direction(self):
        mapping_functions = {
            "minimize": self.map_to_minimize,
            "loss": self.map_to_minimize,
            "maximize": self.map_to_maximize,
            "acc": self.map_to_maximize,
        }

        if self.objectives not in mapping_functions:
            raise ValueError("Invalid objectives")

        return mapping_functions[self.objectives]

    def map_to_minimize(self, results: dict) -> float:
        if "loss" not in results:
            raise ValueError("Invalid results")
        self._objective = "loss"
        self._direction = "minimize"
        result = results["loss"]["metrics"][-1][4]["value"]

        return float(result)

    def map_to_maximize(self, results: dict) -> float:
        if "accuracy" not in results:
            raise ValueError("Invalid results")
        self._objective = "accuracy"
        self._direction = "maximize"
        result = results["accuracy"]["eval_metrics"][0]["value"]
        return float(result)


class OptunaHPO(BaseHPO):
    def __init__(
        self, hub_root: str, hpo_method: str = "RANDOMSAMPLER", direction: str = "maximize"
    ):
        self._study_name = None
        self._hub_root = hub_root
        self._config = HPOMethodConfig("OPTUNA")
        self._hpo_method = hpo_method
        self._objective_direction_mapper = ObjectiveDirectionMapper(direction)
        self._direction = direction
        self._study = None
        self._sampler = None
        self._pruner = None

    def set_study_name(self, study_name: str):
        if study_name is None:
            raise ValueError("Study name cannot be None.")
        self._study_name = study_name

    @property
    def sampler(self):
        return self._sampler

    @property
    def pruner(self):
        return self._pruner

    def _initialize_sampler(self, hpo_method: str, search_space=None) -> None:

        self._sampler, self._pruner = self._config.initialize_method(
            method_type=hpo_method, search_space=search_space
        )

    def create_study(self, search_space: dict) -> None:
        self._initialize_sampler(self._hpo_method, search_space)
        if self._study_name is None:
            raise ValueError("Study name cannot be None.")
        self._study = optuna.create_study(
            study_name=self._study_name,
            storage=f"sqlite:///{self._hub_root}/{self._study_name}/{self._study_name}.db",
            direction=self._direction,
            sampler=self._sampler,
            pruner=self._pruner,
        )
        return {
            "direction": self._direction,
            "sampler": self._sampler.__class__.__name__,
            "pruner": self._pruner.__class__.__name__,
        }

    def visualize_hpo_results(self) -> None:

        param_importance = oplt.plot_param_importances(self._study)
        contour = oplt.plot_contour(self._study)
        coordinates = oplt.plot_parallel_coordinate(self._study)
        slice_plot = oplt.plot_slice(self._study)
        optimization_history = oplt.plot_optimization_history(self._study)

        pio.write_image(param_importance, self._hub_root / self._study_name / "param_importance.png")
        pio.write_image(contour, self._hub_root / self._study_name / "contour.png")
        pio.write_image(coordinates, self._hub_root / self._study_name / "coordinates.png")
        pio.write_image(slice_plot, self._hub_root / self._study_name / "slice_plot.png")
        pio.write_image(
            optimization_history, self._hub_root / self._study_name / "optimization_history.png"
        )

    def optimize(
        self, objective: Callable, dataset: any, n_trials: int, search_space: dict, **kwargs
    ) -> None:
        """
        Runs hyperparameter optimization using Optuna.

        Args:
            study_name (str): The name of the Optuna study.
            objective (str): The name of the objective function to optimize.
            dataset (any): The dataset to use for training.
            n_trials (int): The number of trials to run.
            search_space (dict): The search space for the hyperparameters.
            **kwargs: Additional keyword arguments to pass to the `optimize` method.

        Returns:
            dict: A dictionary containing the best trial number, best parameters, best score, and total time.
        """

        def objective_wrapper(trial: optuna.trial.Trial) -> float:
            """
            A wrapper function for the objective function to optimize.

            Args:
                trial : The Optuna trial object.

            Returns:
                float: The result of the objective function.
            """

            def _get_search_space(trial: optuna.trial.Trial, search_space: dict) -> dict:
                params = {
                    k: trial.suggest_categorical(k, v)
                    if isinstance(v, tuple)
                    else trial.suggest_float(k, v[0], v[1])
                    for k, v in search_space.items()
                }
                return params

            params = _get_search_space(trial, search_space)

            return objective(
                trial=trial,
                dataset=dataset,
                params=params,
                objective_mapper=self._objective_direction_mapper,
                **kwargs,
            )

        self._study.optimize(objective_wrapper, n_trials=n_trials)

    def load_hpo(
        self,
        root_dir: str,
        study_name: str,
    ):
        self.set_study_name = study_name
        self._study = optuna.load_study(
            study_name=self._study_name, storage=f"sqlite:///{root_dir}/{study_name}/{study_name}.db"
        )
        return self._study

    def run_hpo(
        self,
        study_name: str,
        objective: str,
        dataset: any,
        n_trials: int,
        search_space: dict,
        **kwargs,
    ) -> dict:
        """
        Runs hyperparameter optimization using Optuna.

        Args:
            study_name (str): The name of the Optuna study.
            objective (str): The name of the objective function to optimize.
            dataset (any): The dataset to use for training.
            n_trials (int): The number of trials to run.
            search_space (dict): The search space for the hyperparameters.
            **kwargs: Additional keyword arguments to pass to the `optimize` method.

        Returns:
            dict: A dictionary containing the best trial number, best parameters, best score, and total time.
        """
        self.set_study_name(study_name)  # Set the study name using the property setter
        optuna_method_dict = self.create_study(search_space)
        self.optimize(objective, dataset, n_trials, search_space, **kwargs)
        best_value = self._study.best_value
        best_trial = self._study.best_trial.number
        best_params = self._study.best_params
        total_time = str(self._study.trials_dataframe()["duration"].sum())
        return {
            "best_trial": best_trial,
            "best_params": best_params,
            "best_score": best_value,
            "total_time": total_time,
            "methods": optuna_method_dict,
        }
