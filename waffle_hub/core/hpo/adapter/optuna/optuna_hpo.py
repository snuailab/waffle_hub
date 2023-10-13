import gc
import importlib
import inspect
import logging
import os
import warnings
from functools import cached_property
from pathlib import Path, PurePath
from typing import Callable, Union

import optuna
import optuna.visualization as oplt
import plotly.io as pio
from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_hub.core.hpo.adapter.optuna.config import DEFAULT_CONFIG
from waffle_hub.schema.configs import HPOConfig
from waffle_hub.schema.result import HPOResult

from .config import PRUNER_MAP, SAMPLER_MAP
from .hpo_helper import ChoiceMethod, draw_error_image

logger = logging.getLogger(__name__)


class OptunaHPO:
    # directory settings
    DEFAULT_HPO_ROOT_DIR = Path("./hubs")

    HPO_ARTIFACTS_DIR = Path("hpo_artifacts")

    # config files
    CONFIG_DIR = Path("configs")
    HPO_CONFIG_FILE = CONFIG_DIR / "hpo.yaml"

    # hpo results
    HPO_RESULT_FILE = Path("hpo.json")

    DEFAULT_CONFIG = DEFAULT_CONFIG

    def __init__(
        self,
        study_name: str,
        root_dir: Union[str, Path] = None,
        sampler: Union[dict, str] = None,
        pruner: Union[dict, str] = None,
        direction: str = None,
        n_trials: int = None,
        search_space: Union[dict, str] = None,
        metric: str = None,
        is_hub: bool = None,
    ):

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
    def hpo_config_dir(self) -> Path:
        """HPO Config Directory"""
        return self.hpo_dir / OptunaHPO.CONFIG_DIR

    @cached_property
    def hpo_artifacts_dir(self) -> Path:
        """HPO Artifacts Directory"""
        return self.hpo_dir / OptunaHPO.HPO_ARTIFACTS_DIR

    @cached_property
    def hpo_default_config(self) -> Path:
        """HPO Artifacts Directory"""
        return OptunaHPO.HPO_DEFAULT

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
            self.__n_trials = OptunaHPO.DEFAULT_CONFIG.n_trials
            warnings.warn("HPO n_trials is not set. Set to 100.")
        else:
            self.__n_trials = v

    @property
    def search_space(self) -> dict:
        """HPO search_space"""
        return self.__search_space

    @search_space.setter
    def search_space(self, v):
        if isinstance(v, (str, PurePath)):
            # chck if it is yaml or json
            if Path(v).exists():
                if Path(v).suffix in [".yaml", ".yml"]:
                    self.__search_space = io.load_yaml(v)
                elif Path(v).suffix in [".json"]:
                    self.__search_space = io.load_json(v)
                else:
                    raise ValueError(f"search space file should be yaml or json {v}")
            else:
                raise ValueError(f"search space file does not exist {v}")
        elif isinstance(v, dict):
            self.__search_space = v
        elif not isinstance(v, dict):
            raise ValueError(f"search space should be dict or file path {v}")

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
        if v is None:  # if is_hub is not set, check if metric is set (metric is required for hub)
            self.__is_hub = self.__metric is not None
        else:
            self.__is_hub = v

    @property
    def sampler_name(self) -> str:
        """HPO sampler name"""
        return self.__sampler_name

    @sampler_name.setter
    @type_validator(str)
    def sampler_name(self, v):
        if v is None:
            self.__sampler_name = OptunaHPO.DEFAULT_CONFIG.sampler
            warnings.warn("HPO sampler name is not set. Set to TPESampler.")
        else:
            self.__sampler_name = v

    @property
    def sampler_param(self) -> dict:
        """HPO sampler param"""
        return self.__sampler_param

    @sampler_param.setter
    def sampler_param(self, v):
        if v is None:
            self.__sampler_param = None
            warnings.warn("HPO sampler param is not set. Set to {}.")
        else:
            self.__sampler_param = v

    @property
    def pruner_name(self) -> str:
        """HPO pruner name"""
        return self.__pruner_name

    @pruner_name.setter
    @type_validator(str)
    def pruner_name(self, v):
        if v is None:
            self.__pruner_name = OptunaHPO.DEFAULT_CONFIG.pruner
            warnings.warn("HPO pruner name is not set. Set to NopPruner.")
        else:
            self.__pruner_name = v

    @property
    def pruner_param(self) -> dict:
        """HPO pruner param"""
        return self.__pruner_param

    @pruner_param.setter
    def pruner_param(self, v):
        if v is None:
            self.__pruner_param = {}
            warnings.warn("HPO pruner param is not set. Set to {}.")
        else:
            self.__pruner_param = v

    @classmethod
    def parse_root_dir(cls, v):
        if v:
            return Path(v)
        else:
            return cls.DEFAULT_HPO_ROOT_DIR

    @classmethod
    def get_default_hpo_config(cls):
        return cls.DEFAULT_CONFIG

    def save_hpo_config(self):
        HPOConfig(
            sampler=self.sampler_name,
            pruner=self.pruner_name,
            metric=self.metric,
            direction=self.direction,
            n_trials=self.n_trials,
            search_space=self.search_space if self.search_space else {},
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
            return None, None
        return method_name, method_params

    def get_hpo_config(self) -> HPOConfig:
        """Get hpo config from hpo config file.

        Returns:
            HPOConfig: hpo config
        """
        return HPOConfig.load(self.hpo_config_file)

    def _save_hpo_result(self, hpo_results: HPOResult) -> None:
        if self.is_hub:
            best_hpo_root_dir = self.hpo_dir / "hpo" / f"trial_{hpo_results['best_trial']}"
            io.copy_file(best_hpo_root_dir / "configs" / "train.yaml", self.hpo_config_dir)

            for file_name in ["evaluate.json", "metrics.json", "train.py"]:
                io.copy_file(best_hpo_root_dir / file_name, self.hpo_dir)

        hpo_results.save_json(self.hpo_dir / "hpo.json")

    def get_hpo_method(self, method: dict, **hpo_method_params):
        sampler_module = importlib.import_module(method["import_path"])
        sampler_class_name = method["class_name"]

        if sampler_class_name == "NopPruner":
            # 'wrapper_descriptor' object has no attribute '__code__'
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
        method = SAMPLER_MAP[sampler_name.lower()]
        sampler_instance = self.get_hpo_method(method, **kwargs)
        return sampler_instance

    def get_pruner(self, pruner_name: str, **kwargs):
        method = PRUNER_MAP[pruner_name.lower()]
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
        if self.study_name is None:
            raise ValueError("Study name cannot be None.")
        self._study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_name,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    def draw_hpo_plot(self, plot_name: str, study: optuna.Study = None, **kwargs) -> None:
        if study is None:
            study = self._study
        try:
            pio.write_image(
                getattr(oplt, plot_name)(self._study, **kwargs),
                self.hpo_artifacts_dir / f"{plot_name}.png",
            )
        except RuntimeError as e:
            warnings.warn(f"{plot_name} : {str(e)}")
            draw_error_image(
                message=str(e), image_path=str(self.hpo_artifacts_dir / f"{plot_name}.png")
            )

    def visualize_hpo_results(self) -> None:
        if not self.hpo_artifacts_dir.exists():
            io.make_directory(self.hpo_artifacts_dir)
        for visualize in [
            "plot_param_importances",
            "plot_contour",
            "plot_parallel_coordinate",
            "plot_slice",
            "plot_optimization_history",
        ]:
            self.draw_hpo_plot(visualize)

    def _get_search_space(self, trial: optuna.trial.Trial, search_space: dict, is_hub: bool) -> dict:
        hub_params = {}
        params = {}

        for k, v in search_space.items():
            choice_name = k
            method_name = v["method"]
            choices = v["search_space"]
            kwargs = v["kwargs"]

            choice_value = ChoiceMethod(
                choice_name=choice_name, method_name=method_name, choices=choices, **kwargs
            )(trial)

            if k in ["batch_size", "image_size", "learning_rate", "letter_box", "epochs"]:
                hub_params[k] = choice_value
            else:
                params[k] = choice_value

        if is_hub:
            hub_params.update({"advance_params": params})
            return hub_params

        return params

    def optimize(self, objective: Callable, **kwargs) -> None:
        def objective_wrapper(trial: optuna.trial.Trial) -> float:
            search_space_values = self._get_search_space(trial, self.search_space, self.is_hub)

            if self.is_hub:
                search_space_values.update({"metric": self.metric, "trial": trial})

            kwargs.update(search_space_values)
            return objective(**kwargs)

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
        visualize_hpo: bool = True,
        **kwargs,
    ) -> HPOResult:
        """
        Run hyperparameter optimization (HPO) using Optuna.

        Args:
            objective (Callable): The objective function to optimize.
            visualize_hpo (bool, optional): Whether to visualize the HPO results. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the Optuna optimize function.

        Returns:
            HPOResult: An object containing the results of HPO.
        """
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
        if visualize_hpo:
            self.visualize_hpo_results()
        return hpo_results
