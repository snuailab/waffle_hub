import time
from pathlib import Path

import optuna
import optuna.visualization as oplt
import plotly.io as pio
from optuna.pruners import HyperbandPruner
from optuna.samplers import GridSampler, RandomSampler, TPESampler

from waffle_hub.schema.configs import OptunaHpoMethodConfig

# TODO [Exception]: scheduler 및 pruner 관련 none 일 경우 (config 에서도 error 가 발생할 수 있음) 하위 계층에서도 error 발생
#      class HPOMethodError(Exception) 정의 필요


class OptunaHPO:
    def __init__(self, hpo_method):
        self._config = OptunaHpoMethodConfig()
        self._hpo_method = hpo_method
        self._study = None

    # TODO : user can customize sheduler / pruner After update config add property and setter

    def _initialize_sampler(self, hpo_method):
        self._sampler, self._pruner = self._config.initialize_method(hpo_method)

    # TODO : hub / create_study
    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
    ):
        self._initialize_sampler(self._hpo_method)
        self._study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{study_name}.db",
            direction=direction,
            sampler=self._sampler,
            pruner=self._pruner,
        )

    def load_study(
        self,
        study_name: str,
    ):
        # load studies using db
        self._study = optuna.load_study(study_name, storage=f"sqlite:///{study_name}.db")
        # db must be located in study hub
        pass

    def visualization(self, dataset_path: Path):
        # 시각화 생성
        param_importance = oplt.plot_param_importances(self._study)
        contour = oplt.plot_contour(self._study)
        coordinates = oplt.plot_parallel_coordinate(self._study)
        slice_plot = oplt.plot_slice(self._study)
        optimization_history = oplt.plot_optimization_history(self._study)

        # 생성한 시각화를 이미지 파일로 저장
        pio.write_image(param_importance, dataset_path / "param_importance.png")
        pio.write_image(contour, dataset_path / "contour.png")
        pio.write_image(coordinates, dataset_path / "coordinates.png")
        pio.write_image(slice_plot, dataset_path / "slice_plot.png")
        pio.write_image(optimization_history, dataset_path / "optimization_history.png")

    def optimize(self, objective, dataset, n_trials, search_space, **kwargs):
        def objective_wrapper(trial):
            def _get_search_space(trial, search_space):
                params = {}
                for k, v in search_space.items():
                    if isinstance(v[0], bool):
                        params[k] = trial.suggest_categorical(k, v)
                    else:
                        params[k] = trial.suggest_uniform(k, v[0], v[1])
                return params

            params = _get_search_space(trial, search_space)
            return objective(trial=trial, dataset=dataset, params=params, **kwargs)

        self._study.optimize(objective_wrapper, n_trials=n_trials)

    def _create_hpo(
        self, study_name, objective, dataset, n_trials, direction, search_space, **kwargs
    ):
        self.create_study(study_name=study_name, direction=direction)
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
        }
