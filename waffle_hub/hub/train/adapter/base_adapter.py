from abc import abstractmethod
from pathlib import Path
from typing import Union

from waffle_hub import TaskType
from waffle_hub.hub.model.base_model import Model
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.hub.train.base_trainer import Trainer
from waffle_hub.schema.fields.category import Category


class BaseAdapter(Model, Trainer):
    """
    Base Train Adapter
    """

    # abstract property
    ## model spec
    BACKEND_NAME = None
    VERSION = None
    MODEL_TYPES = None

    ## trainer spec
    MULTI_GPU_TRAIN = None
    DEFAULT_PARAMS = None
    DEFAULT_ADVANCE_PARAMS = None

    def __init__(
        self,
        hub_dir: Path,
        name: str,
        task: Union[str, TaskType],
        model_type: str,
        model_size: str,
        categories: list[Union[str, int, float, dict, Category]],
    ):
        Model.__init__(
            self,
            hub_dir=hub_dir,
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
        )
        Trainer.__init__(
            self,
            hub_dir=hub_dir,
        )
        # abstract property
        if self.BACKEND_NAME is None:
            raise AttributeError("BACKEND_NAME must be specified.")

        if self.VERSION is None:
            raise AttributeError("VERSION must be specified.")

        if self.MODEL_TYPES is None:
            raise AttributeError("MODEL_TYPES must be specified.")

        if self.MULTI_GPU_TRAIN is None:
            raise AttributeError("MULTI_GPU_TRAIN must be specified.")

        if self.DEFAULT_PARAMS is None:
            raise AttributeError("DEFAULT_PARAMS must be specified.")

    ##--
    def get_default_advance_train_params(
        cls, task: str = None, model_type: str = None, model_size: str = None
    ) -> dict:
        return cls.DEFAULT_ADVANCE_PARAMS

    # Model abstract method
    @abstractmethod
    def get_model(self) -> ModelWrapper:
        raise NotImplementedError

    @abstractmethod
    def get_preprocess(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_postprocess(self, *args, **kwargs):
        raise NotImplementedError

    # Trainer abstract method
    @abstractmethod
    def training(self):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError
