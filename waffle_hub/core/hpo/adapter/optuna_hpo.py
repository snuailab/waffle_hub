import importlib
import inspect
import logging
import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import Callable, Union

import optuna
import optuna.visualization as oplt
import plotly.io as pio
from optuna.pruners import NopPruner
from optuna.samplers import RandomSampler
from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_hub import PRUNER_MAP, SAMPLER_MAP

# from waffle_hub.core.hpo.base_hpo import BaseHPO
from waffle_hub.schema.configs import HPOConfig
from waffle_hub.schema.result import HPOResult

logger = logging.getLogger(__name__)


class OptunaHPO:
    # DEFAULT_HPO_PARAMS = None
    # directory settings
    DEFAULT_HPO_ROOT_DIR = Path("./hubs")

    HPO_ARTIFACTS_DIR = Path("hpo_artifacts")

    # config files
    CONFIG_DIR = Path("configs")
    HPO_CONFIG_FILE = CONFIG_DIR / "hpo.yaml"

    # hpo results
    HPO_RESULT_FILE = Path("hpo.json")

    def __init__(
        self,
        study_name: str,
        root_dir: Union[str, Path] = None,
        sampler: Union[dict, str] = None,
        pruner: Union[dict, str] = None,
        direction: str = None,
        n_trials: int = None,
        search_space: dict = None,
        metric: str = None,
        is_hub: bool = None,
    ):
        # if self.DEFAULT_HPO_PARAMS is None:
        #     raise AttributeError("DEFAULT_HPO_PARAMS is not set.")

        # hpo study configuration
        self.study_name: str = study_name
        self.root_dir: Path = root_dir

        # Extract sampler name and kwargs
        self.sampler_name, self.sampler_param = self.extract_method_name_and_parms(sampler)

        # Extract pruner name and kwargs
        self.pruner_name, self.pruner_param = self.extract_method_name_and_parms(pruner)

        self.direction = direction
        self.n_trials = n_trials
        self.search_space = search_space
        self.metric = metric
        self.is_hub = is_hub
        self._study = None
        self.save_hpo_config()

    def __repr__(self):
        return self.get_hpo_config().__repr__()

    @cached_property
    def hpo_dir(self) -> Path:
        """HPO Directory"""
        return self.root_dir / self.study_name

    @cached_property
    def hpo_config_file(self) -> Path:
        """HPO Config yaml File"""
        return self.hpo_dir / OptunaHPO.HPO_CONFIG_FILE

    @cached_property
    def hpo_result_file(self) -> Path:
        """HPO Result json File"""
        return self.hpo_dir / OptunaHPO.HPO_RESULT_FILE

    @cached_property
    def hpo_artifacts_dir(self) -> Path:
        """HPO Artifacts Directory"""
        return self.hpo_dir / OptunaHPO.HPO_ARTIFACTS_DIR

    @cached_property
    def storage_name(self) -> str:
        """HPO Storage Name"""
        return f"sqlite:///{self.root_dir}/{self.study_name}/{self.study_name}.db"

    # properties
    @property
    def study_name(self) -> str:
        """HPO Study Name"""
        return self.__study_name

    @study_name.setter
    @type_validator(str)
    def study_name(self, v):
        self.__study_name = v

    @property
    def root_dir(self) -> Path:
        """Root Directory"""
        return self.__root_dir

    @root_dir.setter
    @type_validator(Path, strict=False)
    def root_dir(self, v):
        self.__root_dir = OptunaHPO.parse_root_dir(v)
        logger.info(f"HPO root directory : {self.root_dir}")

    @property
    def direction(self) -> str:
        """HPO direction"""
        return self.__direction

    @direction.setter
    @type_validator(str)
    def direction(self, v):
        if v is None:
            warnings.warn("HPO direction is not set. Set to maximize or minimize.")
        self.__direction = v

    @property
    def n_trials(self) -> int:
        """HPO n_trials"""
        return self.__n_trials

    @n_trials.setter
    @type_validator(int)
    def n_trials(self, v):
        if v is None:
            warnings.warn("HPO n_trials is not set. Set to 100.")
        self.__n_trials = v

    @property
    def search_space(self) -> dict:
        """HPO search_space"""
        return self.__search_space

    @search_space.setter
    @type_validator(dict)
    def search_space(self, v):
        if v is None:
            warnings.warn("HPO search_space is not set. Set to None.")
        self.__search_space = v

    @property
    def metric(self) -> str:
        """HPO metric"""
        return self.__metric

    @metric.setter
    @type_validator(str)
    def metric(self, v):
        if v is None:
            warnings.warn("HPO metric is not set. Set to None.")
        self.__metric = v

    @property
    def is_hub(self) -> bool:
        """Is Hub"""
        return self.__is_hub

    @is_hub.setter
    @type_validator(bool)
    def is_hub(self, v):
        if v == None:  # if is_hub is not set, check if metric is set (metric is required for hub)
            self.__is_hub = self.__metric is not None
        else:
            self.__is_hub = v

    @classmethod
    def parse_root_dir(cls, v):
        if v:
            return Path(v)
        else:
            return cls.DEFAULT_HPO_ROOT_DIR

    def save_hpo_config(self):
        HPOConfig(
            sampler=self.sampler_name,
            pruner=self.pruner_name,
            metric=self.metric,
            direction=self.direction,
            n_trials=self.n_trials,
            search_space=self.search_space,
        ).save_yaml(self.hpo_config_file)

    def get_hpo_config(self) -> HPOConfig:
        """Get hpo config from hpo config file.

        Returns:
            HPOConfig: hpo config
        """
        return HPOConfig.load(self.hpo_config_file)

    def extract_method_name_and_parms(self, value):
        if isinstance(value, dict):
            method_name = next(iter(value))
            method_params = value[method_name]
        elif isinstance(value, str):
            method_name = value
            method_params = {}
        else:
            raise ValueError(
                f"Invalid value for method {value}\
                             value must be a dict or a str."
            )
        return method_name, method_params

    # def get_default_hpo_config(self):
    #     return HPOConfig(
    #         sampler="randomsampler",
    #         pruner="nopruner",
    #         direction="maximize",
    #         n_trials=100,
    #     )

    def get_hpo_config(self) -> HPOConfig:
        """Get hpo config from hpo config file.

        Returns:
            HPOConfig: hpo config
        """
        return HPOConfig.load(self.hpo_config_file)

    def _save_hpo_result(self, hpo_results: HPOResult) -> None:
        if self.is_hub:
            best_hpo_root_dir = self.hpo_dir / "hpo" / f"trial_{hpo_results['best_trial']}"

            io.copy_file(best_hpo_root_dir / "configs" / "train.yaml", self.CONFIG_DIR)

            for file_name in ["evaluate.json", "metrics.json", "train.py"]:
                io.copy_file(best_hpo_root_dir / file_name, self.hpo_dir)

        hpo_results.save_json(self.hpo_dir / "hpo.json")

    def get_hpo_method(self, method: dict, **hpo_method_params):
        sampler_module = importlib.import_module(method["import_path"])
        sampler_class_name = method["class_name"]

        if (
            sampler_class_name == "NopPruner"
        ):  # 'wrapper_descriptor' object has no attribute '__code__'
            return getattr(sampler_module, sampler_class_name)

        if not hasattr(sampler_module, sampler_class_name):
            raise ValueError(
                f"Sampler class {sampler_class_name} not found in module {sampler_module}"
            )
        sampler_class = getattr(sampler_module, sampler_class_name)
        # Check if all required arguments are present
        required_args = set(sampler_class.__init__.__code__.co_varnames[1:])
        given_args = set(hpo_method_params.keys())
        for arg in given_args:
            if arg not in required_args:
                raise ValueError(
                    f"Argument {arg} not found in sampler {sampler_class_name} \
                                 \nRequired arguments are {required_args}"
                )
        try:
            sampler_instance = sampler_class(**hpo_method_params)
        except TypeError as e:
            raise TypeError(
                f"Error while initializing sampler {sampler_class_name} with arguments {kwargs}"
            ) from e

        return sampler_instance

    def get_sampler(self, sampler_name: str, **kwargs):
        method = SAMPLER_MAP[sampler_name]
        sampler_instance = self.get_hpo_method(method, **kwargs)
        return sampler_instance

    def get_pruner(self, pruner_name: str, **kwargs):
        method = PRUNER_MAP[pruner_name]
        pruner_instance = self.get_hpo_method(method, **kwargs)
        return pruner_instance

    def get_method_params(self, method):
        sampler_module = importlib.import_module(method["import_path"])
        sampler_class_name = method["class_name"]
        if not hasattr(sampler_module, sampler_class_name):
            raise ValueError(
                f"Sampler class {sampler_class_name} not found in module {sampler_module}"
            )
        sampler_class = getattr(sampler_module, sampler_class_name)
        # Check if all required arguments are present
        required_args = set(sampler_class.__init__.__code__.co_varnames[1:])
        return required_args

    def get_pruner_params(self, pruner_name: str):
        method = PRUNER_MAP[pruner_name]
        if method["class_name"] == "NopPruner":
            return None
        return self.get_method_params(method)

    def get_sampler_params(self, sampler_name: str):
        method = SAMPLER_MAP[sampler_name]
        return self.get_method_params(method)

    def create_study(self, sampler, pruner) -> None:
        # sampler, pruner = self._initialize_sampler("BOHB", search_space)
        if self.study_name is None:
            raise ValueError("Study name cannot be None.")

        # # make study directory
        # if not os.path.exists(self.hpo_dir):
        #     io.make_directory(self.hpo_dir)

        self._study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_name,
            direction=self.direction,
            sampler=RandomSampler(),
            pruner=NopPruner(),
        )

    def visualize_hpo_results(self) -> None:
        if not os.path.exists(self.hpo_artifacts_dir):
            io.make_directory(self.hpo_artifacts_dir)
        pio.write_image(
            oplt.plot_param_importances(self._study), self.hpo_artifacts_dir / "param_importance.png"
        )
        pio.write_image(oplt.plot_contour(self._study), self.hpo_artifacts_dir / "contour.png")
        pio.write_image(
            oplt.plot_parallel_coordinate(self._study), self.hpo_artifacts_dir / "coordinates.png"
        )
        pio.write_image(oplt.plot_slice(self._study), self.hpo_artifacts_dir / "slice_plot.png")
        pio.write_image(
            oplt.plot_optimization_history(self._study),
            self.hpo_artifacts_dir / "optimization_history.png",
        )

    def optimize(self, objective: Callable, **kwargs) -> None:
        def objective_wrapper(trial: optuna.trial.Trial) -> float:
            def _get_search_space(trial: optuna.trial.Trial, search_space: dict) -> dict:
                if "advance_params" in search_space:
                    advance_params = search_space.pop("advance_params")
                    advance_params = {
                        k: trial.suggest_categorical(k, v)
                        if isinstance(v, tuple)
                        else trial.suggest_float(k, v[0], v[1])
                        for k, v in advance_params.items()
                    }
                    params.update({"advance_params": advance_params})
                params = {
                    k: trial.suggest_categorical(k, v)
                    if isinstance(v, tuple)
                    else trial.suggest_float(k, v[0], v[1])
                    for k, v in search_space.items()
                }
                return params

            params = _get_search_space(trial, self.search_space)
            kwargs.update(params)
            kwargs.update({"trial": trial})
            kwargs.update({"metric": self.metric})
            return objective(
                **kwargs,
            )

        self._study.optimize(objective_wrapper, n_trials=self.n_trials)

    def load_hpo(
        self,
        root_dir: str,
        study_name: str,
    ):
        self._study = optuna.load_study(
            study_name=self._study_name, storage=f"sqlite:///{root_dir}/{study_name}/{study_name}.db"
        )
        return self._study

    def run_hpo(
        self,
        objective: Callable,
        **kwargs,
    ) -> HPOResult:
        sampler = self.get_sampler(self.sampler_name, **self.sampler_param)
        pruner = self.get_pruner(self.pruner_name, **self.pruner_param)
        self.create_study(sampler, pruner)
        self.optimize(objective=objective, **kwargs)
        hpo_results = HPOResult(
            best_trial=self._study.best_trial.number,
            best_params=self._study.best_params,
            best_score=self._study.best_value,
            total_time=str(self._study.trials_dataframe()["duration"].sum()),
        )
        self._save_hpo_result(hpo_results)
        self.visualize_hpo_results()
        return hpo_results
