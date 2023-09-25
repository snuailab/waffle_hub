from abc import ABC, abstractmethod
from typing import Callable


class BaseHPO(ABC):
    """
    Base class for hyperparameter optimization.
    """

    @abstractmethod
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
        Runs hyperparameter optimization.

        Args:
            study_name (str): The name of the study.
            objective (str): The name of the objective function to optimize.
            dataset (any): The dataset to use for training.
            n_trials (int): The number of trials to run.
            search_space (dict): The search space for the hyperparameters.
            **kwargs: Additional keyword arguments to pass to the `optimize` method.

        Returns:
            dict: A dictionary containing the best trial number, best parameters, best score, and total time.
        """
        # not implemented error 를 추가해줘야함
        raise NotImplementedError("run_hpo method is not implemented in the inherited class.")

    @abstractmethod
    def optimize(
        self, objective: Callable, dataset: any, n_trials: int, search_space: dict, **kwargs
    ) -> None:
        """
        Optimizes hyperparameters.

        Args:
            objective (Callable): The objective function to optimize.
            dataset (any): The dataset to use for training.
            n_trials (int): The number of trials to run.
            search_space (dict): The search space for the hyperparameters.
            **kwargs: Additional keyword arguments to pass to the `objective` function.
        """
        raise NotImplementedError("run_hpo method is not implemented in the inherited class.")
