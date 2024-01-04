import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from torch import nn
from waffle_utils.utils import type_validator

from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import ModelConfig
from waffle_hub.schema.fields.category import Category
from waffle_hub.type import TaskType


class Model(ABC):
    """
    Base class for training manager
    """

    # Model spec, abstract property
    BACKEND_NAME = None
    VERSION = None
    MODEL_TYPES = None

    # directory settting
    CONFIG_DIR = Path("config")

    # train config file name
    MODEL_CONFIG_FILE = "model.yaml"

    def __init__(
        self,
        root_dir: Path,
        name: str,
        task: Union[str, TaskType],
        model_type: str,
        model_size: str,
        categories: list[Union[str, int, float, dict, Category]],
        load: bool = False,
    ):
        # abstract property
        if self.BACKEND_NAME is None:
            raise AttributeError("BACKEND_NAME must be specified.")

        if self.VERSION is None:
            raise AttributeError("VERSION must be specified.")

        if self.MODEL_TYPES is None:
            raise AttributeError("MODEL_TYPES must be specified.")

        self.root_dir = root_dir
        self.name = name
        self.task = task
        self.model_type = model_type
        self.model_size = model_size
        self.categories = categories

        if self.model_config_file.exists() and not load:
            raise FileExistsError("Model already exists. Try to 'load_manager' function.")

        self.save_model_config(
            model_config_file=self.model_config_file,
        )

    # properties
    @property
    def task(self) -> str:
        """Task Name"""
        return self.__task

    @task.setter
    def task(self, v):
        if v not in list(self.MODEL_TYPES.keys()):
            raise ValueError(
                f"Task {v} is not supported. Choose one of {list(self.MODEL_TYPES.keys())}"
            )
        self.__task = str(v.value) if isinstance(v, TaskType) else str(v)

    @property
    def model_type(self) -> str:
        """Model Type"""
        return self.__model_type

    @model_type.setter
    @type_validator(str)
    def model_type(self, v):
        if v not in self.MODEL_TYPES[self.task]:
            raise ValueError(
                f"Model Type {v} is not supported. Choose one of {self.MODEL_TYPES[self.task]}"
            )
        self.__model_type = v

    @property
    def model_size(self) -> str:
        """Model Size"""
        return self.__model_size

    @model_size.setter
    @type_validator(str)
    def model_size(self, v):
        if v not in self.MODEL_TYPES[self.task][self.model_type]:
            raise ValueError(
                f"Model Size {v} is not supported. Choose one of {self.MODEL_TYPES[self.task][self.model_type]}"
            )
        self.__model_size = v

    @property
    def version(self) -> str:
        """Version"""
        return self.__version

    @version.setter
    @type_validator(str)
    def version(self, v):
        self.__version = v

    @property
    def categories(self) -> list[Category]:
        return self.__categories

    @categories.setter
    @type_validator(list)
    def categories(self, v):
        if v is None or len(v) == 0:
            warnings.warn(
                "Categories is not specified.\n"
                + "It follows the categories of Dataset when the training starts."
            )
            v = []
        elif isinstance(v[0], dict):
            v = [
                getattr(Category, self.task.lower())(
                    **{
                        **category,
                        "category_id": category.get("category_id", i),
                    }
                )
                for i, category in enumerate(v, start=1)
            ]
        elif isinstance(v[0], (str, int, float)):
            v = [
                getattr(Category, self.task.lower())(
                    category_id=i,
                    supercategory="object",
                    name=str(category),
                )
                for i, category in enumerate(v, start=1)
            ]
            warnings.warn(
                "Super category is not specified. It may cause unexpected errors in some backends.\n"
                + "To avoid this warning, please specify category as a list of dictionary or Category"
            )
        elif isinstance(v[0], Category):
            pass

        self.__categories = v

    @property
    def config_dir(self) -> Path:
        """Config Directory"""
        return self.root_dir / self.CONFIG_DIR

    @property
    def model_config_file(self) -> Path:
        """Model Config yaml File"""
        return self.config_dir / self.MODEL_CONFIG_FILE

    # TODO: implement
    # @abstractmethod
    # def load_model(self, model_path: Path) -> nn.Module:
    #     # load model from model_path
    #     # need to set self.model
    #     raise NotImplementedError

    @abstractmethod
    def _get_preprocess(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_postprocess(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> ModelWrapper:
        """Get model for inference or evaluation
        Returns:
            ModelWrapper: best model wrapper
        """
        raise NotImplementedError

        # # get adapt functions
        # preprocess = self.preprocess()
        # postprocess = self.postprocess()

        # # return model wrapper
        # return ModelWrapper(
        #     model=self.model.eval(),
        #     preprocess=preprocess,
        #     postprocess=postprocess,
        # )

    @classmethod
    def get_model_config(cls, root_dir: Union[str, Path]) -> ModelConfig:
        """Get model config from model config yaml file

        Args:
            root_dir (Path): root directory of model config yaml file

        Returns:
            ModelConfig: model config
        """
        model_config_file_path = Path(root_dir) / cls.CONFIG_DIR / cls.MODEL_CONFIG_FILE
        if not model_config_file_path.exists():
            warnings.warn(f"Model config file {model_config_file_path} is not exist.")
            return []
        return ModelConfig.load(model_config_file_path)

    def save_model_config(
        self,
        model_config_file: Path,
    ):
        """Save model config to model config yaml file

        Args:
            model_config_file (Path): model config yaml file
        """
        ModelConfig(
            name=self.name,
            backend=self.BACKEND_NAME,
            version=self.VERSION,
            task=self.task,
            model_type=self.model_type,
            model_size=self.model_size,
            categories=list(map(lambda x: x.to_dict(), self.categories)),
        ).save_yaml(model_config_file)

    # def load_model_config(self, model_config_file: Path):
    #     """Load model config from model config yaml file (set self.model_cfg from yaml file)

    #     Args:
    #         model_config_file (Path): model config yaml file
    #     """
    #     self.model_cfg = ModelConfig.load(model_config_file)
    def set_model_name(self, name: str):
        """Set model name
        if model name is not same with model config name, it will cause unexpected errors

        Args:
            name (str): model name
        """
        self.name = name
        ModelConfig(
            name=self.name,
            backend=self.BACKEND_NAME,
            version=self.VERSION,
            task=self.task,
            model_type=self.model_type,
            model_size=self.model_size,
            categories=list(map(lambda x: x.to_dict(), self.categories)),
        ).save_yaml(self.model_config_file)

    def get_categories(self) -> list[Category]:
        return self.categories

    def get_category_names(self) -> list[str]:
        return [category.name for category in self.categories]
