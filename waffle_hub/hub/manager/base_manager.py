import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Union

from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_dough.type.task_type import TaskType
from waffle_hub import EXPORT_MAP
from waffle_hub.dataset import Dataset
from waffle_hub.hub.model.base_model import Model
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.hub.train.base_trainer import Trainer
from waffle_hub.schema.configs import ModelConfig, TrainConfig
from waffle_hub.schema.fields.category import Category


class BaseManager:
    """
    Base Manager (Train, Model)
    """

    # abstract property
    ## model spec
    BACKEND_NAME = None
    BACKEND_VERSION = None
    MODEL_TYPES = None

    ## trainer spec
    MULTI_GPU_TRAIN = None
    DEFAULT_PARAMS = None
    DEFAULT_ADVANCE_PARAMS = None

    # directory settting
    CONFIG_DIR = Path("config")

    # train config file name
    MODEL_CONFIG_FILE = "model.yaml"
    TRAIN_CONFIG_FILE = "train.yaml"

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
        # abstract property (Model)
        if self.BACKEND_NAME is None:
            raise AttributeError("BACKEND_NAME must be specified.")

        if self.BACKEND_VERSION is None:
            raise AttributeError("VERSION must be specified.")

        if self.MODEL_TYPES is None:
            raise AttributeError("MODEL_TYPES must be specified.")

        # abstract property (Trainer)
        if self.MULTI_GPU_TRAIN is None:
            raise AttributeError("MULTI_GPU_TRAIN must be specified.")

        if self.DEFAULT_PARAMS is None:
            raise AttributeError("DEFAULT_PARAMS must be specified.")

        self.root_dir = root_dir
        self.name = name
        self.task = task
        self.model_type = model_type
        self.model_size = model_size
        self.categories = categories

        self.backend = self.BACKEND_NAME

        if self.model_config_file.exists() and not load:
            raise FileExistsError("Model already exists. Try to 'load_manager' function.")

        self.save_model_config(
            model_config_file=self.model_config_file,
        )
        self.trainer = self._init_trainer()

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

    @property
    def train_config_file(self) -> Path:
        """Train Config yaml File"""
        return self.config_dir / self.TRAIN_CONFIG_FILE

    # Model abstract method
    @abstractmethod
    def get_model(self) -> ModelWrapper:
        """Get model for inference or evaluation
        Returns:
            ModelWrapper: best model wrapper
        """
        raise NotImplementedError

    @abstractmethod
    def _get_preprocess(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_postprocess(self, *args, **kwargs):
        raise NotImplementedError

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

    @classmethod
    def get_train_config(cls, root_dir: Union[str, Path]) -> TrainConfig:
        """Get train config from train config yaml file.

        Args:
            root_dir (Path): root directory of train config yaml file

        Returns:
            TrainConfig: train config of train config yaml file
        """
        train_config_file_path = Path(root_dir) / cls.CONFIG_DIR / cls.TRAIN_CONFIG_FILE
        if not train_config_file_path.exists():
            warnings.warn(f"Train config file {train_config_file_path} is not exist. Train first!")
            return None
        return TrainConfig.load(train_config_file_path)

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

    def save_train_config(
        self,
        cfg: TrainConfig,
        train_config_path: Path,
    ):
        """Save train config to yaml file

        Args:
            train_config_path (Path): file path for saving train config
        """
        cfg.save_yaml(train_config_path)

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

    # Trainer abstract method
    @abstractmethod
    def _init_trainer(self):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError

    def check_train_sanity(self) -> bool:
        """Check if all essential files are exist.

        Returns:
            bool: True if all files are exist else False
        """

    @classmethod
    def is_exists(cls, root_dir: str) -> bool:
        """
        Manager is exists (model config file is exists)

        Args:
            root_dir (str): Root directory

        Returns:
            bool: True or False
        """
        model_config_file = Path(root_dir) / cls.CONFIG_DIR / cls.MODEL_CONFIG_FILE
        return model_config_file.exists()

    @classmethod
    def from_model_config_file(
        cls, root_dir: str, name: str, model_config_file_path: Union[str, Path]
    ):
        """
        Create Manager from model config file

        Args:
            root_dir (str): Root directory
            name (str): Model name
            model_config_file_path (Union[str, Path]): Model config file path

        raises:
            ValueError: If model backend is not matched with hub backend

        Returns:
            BaseManager: Manager
        """
        if not model_config_file_path.exists():
            raise FileNotFoundError(f"Model config file {model_config_file_path} is not exist.")

        model_config = ModelConfig.load(model_config_file_path)

        if model_config["backend"] != cls.BACKEND_NAME:
            raise ValueError(
                f"Model backend is not matched with hub backend. Model backend: {model_config['backend']}, Hub backend: {cls.BACKEND_NAME}"
            )

        return cls(
            root_dir=root_dir,
            name=name,
            task=model_config["task"],
            model_type=model_config["model_type"],
            model_size=model_config["model_size"],
            categories=model_config["categories"],
        )

    @classmethod
    def load(cls, root_dir: str):
        """
        Load Manager from model config file in root directory

        Args:
            root_dir (str): Root directory

        Returns:
            BaseManager: Manager
        """
        model_config_file_path = Path(root_dir) / cls.CONFIG_DIR / cls.MODEL_CONFIG_FILE
        if not model_config_file_path.exists():
            raise FileNotFoundError(
                f"Model config file {model_config_file_path} is not exist. Init first."
            )

        model_config = ModelConfig.load(model_config_file_path)

        if model_config["backend"] != cls.BACKEND_NAME:
            raise ValueError(
                f"Model backend is not matched with hub backend. Model backend: {model_config['backend']}, Hub backend: {cls.BACKEND_NAME}"
            )

        return cls(
            root_dir=root_dir,
            name=model_config["name"],
            task=model_config["task"],
            model_type=model_config["model_type"],
            model_size=model_config["model_size"],
            categories=model_config["categories"],
            load=True,
        )

    # need override
    def get_default_advance_train_params(
        cls, task: str = None, model_type: str = None, model_size: str = None
    ) -> dict:
        """
        Get default train advance params

        Args:
            task (str): Task name
            model_type (str): Model type
            model_size (str): Model size

        Raises:
            ModuleNotFoundError: If backend is not supported

        Returns:
            dict: Default train advance params
        """
        raise NotImplementedError(f"{cls.BACKEND_NAME} does not support advance_params argument.")

    # common function
    def delete_manager(self):
        """
        Delete manager.
        """
        # TODO: utils 1.0 연동 시 get 함수 사용
        if self.config_dir.exists():
            io.remove_directory(self.config_dir)
        if self.artifacts_dir.exists():
            io.remove_directory(self.artifacts_dir)
        if self.weights_dir.exists():
            io.remove_directory(self.weights_dir)
        if self.train_log_dir.exists():
            io.remove_directory(self.train_log_dir)
        del self
        return None

    def parse_dataset(
        self, dataset: Union[Dataset, str, Path], dataset_root_dir: str = None
    ) -> (Path, str):
        """parse dataset

        Args:
            dataset (Union[Dataset, str, Path]): Dataset
            dataset_root_dir (str): Dataset root directory

        Returns:
            (Path, Path): Export directory, train dataset path
        """
        #
        if isinstance(dataset, (str, Path)):
            if Path(dataset).exists():
                dataset = Path(dataset)
                dataset = Dataset.load(
                    name=dataset.parts[-1], root_dir=dataset.parents[0].absolute()
                )
            elif dataset in Dataset.get_dataset_list(dataset_root_dir):
                dataset = Dataset.load(name=dataset, root_dir=dataset_root_dir)
            else:
                raise FileNotFoundError(f"Dataset {dataset} is not exist.")

        # check category match
        if not self.categories:
            self.categories = dataset.get_categories()
            self.save_model_config(self.model_config_file)
        elif set(dataset.get_category_names()) != set(self.get_category_names()):
            raise ValueError(
                "Dataset categories are not matched with hub categories. \n"
                + f"Dataset categories: {dataset.get_category_names()}, Hub categories: {self.get_category_names()}"
            )

        # check task match
        if dataset.task.lower() != self.task.lower():
            raise ValueError(
                f"Dataset task is not matched with hub task. Dataset task: {dataset.task}, Hub task: {self.task}"
            )

        # convert dataset to backend format if not exist
        export_dir = dataset.export_dir / EXPORT_MAP[self.backend]
        if not export_dir.exists():
            ## LOGGER!!
            # logger.info(f"[Dataset] Exporting dataset to {self.backend} format...")
            export_dir = dataset.export(self.backend)
            # logger.info("[Dataset] Exporting done.")

        return export_dir, dataset.dataset_dir

    def train(
        self,
        dataset: Union[Dataset, str, Path],
        dataset_root_dir: str = None,
        epochs: int = None,
        batch_size: int = None,
        image_size: Union[int, list[int]] = None,
        learning_rate: float = None,
        letter_box: bool = None,
        pretrained_model: str = None,
        device: str = "0",
        workers: int = 2,
        seed: int = 0,
        advance_params: Union[dict, str] = None,
        verbose: bool = True,
    ) -> TrainResult:
        """Start Train

        Args:
            dataset (Union[Dataset, str, Path]): Waffle Dataset object or path or name.
            dataset_root_dir (str, optional): Waffle Dataset root directory. Defaults to None.
            epochs (int, optional): number of epochs. None to use default. Defaults to None.
            batch_size (int, optional): batch size. None to use default. Defaults to None.
            image_size (Union[int, list[int]], optional): image size. None to use default. Defaults to None.
            learning_rate (float, optional): learning rate. None to use default. Defaults to None.
            letter_box (bool, optional): letter box. None to use default. Defaults to None.
            pretrained_model (str, optional): pretrained model. None to use default. Defaults to None.
            device (str, optional):
                "cpu" or "gpu_id" or comma seperated "gpu_ids". Defaults to "0".
            workers (int, optional): number of workers. Defaults to 2.
            seed (int, optional): random seed. Defaults to 0.
            advance_params (Union[dict, str], optional): advance params dictionary or file (yaml, json) path. Defaults to None.
            verbose (bool, optional): verbose. Defaults to True.

        Returns:
            TrainResult: train result
        """
        return self.trainer.train()
