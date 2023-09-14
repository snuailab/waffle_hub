import time

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import GridSampler, RandomSampler, TPESampler


class OptunaHPO:
    def __init__(self):
        self._scheduler = None
        self._pruner = None
        self._hpo_method = None

    # TODO : hpo_method -> scheduler, pruner
    @property
    def scheduler(self):
        # TODO : exeption
        return self._scheduler

    @property
    def pruner(
        self,
    ):
        # TODO : exeption
        return self._pruner

    @property
    def hpo_method(
        self,
    ):
        # TODO : exeption
        return self._hpo_method

    # user can customize sheduler / pruner
    @scheduler.setter
    def scheduler(self, scheduler: str):
        pass

    @pruner.setter
    def pruner(self, pruner: str):
        pass

    @hpo_method.setter
    def hpo_method(
        self,
        hpo_method: str,
    ):
        # sampler_class, pruner_class = OptunaHPOEnum.get_sampler_and_pruner(hpo_method.upper())
        # if sampler_class is not None:
        #     sampler = sampler_class()
        # else:
        #     print("Sampler not found.")
        pass

    def _create_sampler(self, sampler_type, search_space):
        samplers = {
            "RandomSampler": RandomSampler(),
            "TPESampler": TPESampler(),
            "GridSampler": GridSampler(search_space),
        }
        if sampler_type in samplers:
            return samplers[sampler_type]
        else:
            raise ValueError(f"Invalid sampler type: {sampler_type}")

    def create_study(
        self,
    ):
        pass

    def load_study(
        self,
    ):
        # load studies using db
        # db must be located in study hub
        pass

    def visualization(
        self,
    ):
        pass

    def optimize(
        self, objective, dataset, n_trials, direction, sampler_type, search_space, **kwargs
    ):
        start_time = time.time()
        sampler = self._create_sampler(sampler_type, search_space)
        study = optuna.create_study(direction=direction, sampler=sampler)

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
            return objective(trial, dataset, params, **kwargs)

        study.optimize(objective_wrapper, n_trials=n_trials)
        end_time = time.time()
        best_value = study.best_value
        best_trial = study.best_trial.number
        best_params = study.best_params
        total_time = end_time - start_time
        return best_trial, best_params, best_value, total_time
