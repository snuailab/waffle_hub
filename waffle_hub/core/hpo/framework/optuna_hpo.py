import time

import optuna
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
        self._sampler = None
        self._pruner = None

    # TODO : user can customize sheduler / pruner After update config add property and setter

    def _initialize_sampler(self, hpo_method):
        self._sampler, self._pruner = self._config.initialize_method(hpo_method)

    def create_study(
        self,
        study_name: str = "test",
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
        # db must be located in study hub
        pass

    def visualization(
        self,
    ):
        pass

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

    def hpo(self, objective, dataset, n_trials, direction, search_space, **kwargs):
        start_time = time.time()
        self.create_study(direction=direction)
        self.optimize(objective, dataset, n_trials, search_space)
        end_time = time.time()
        best_value = self._study.best_value
        best_trial = self._study.best_trial.number
        best_params = self._study.best_params
        total_time = end_time - start_time
        return best_trial, best_params, best_value, total_time
