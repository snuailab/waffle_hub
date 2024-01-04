from abc import abstractmethod
from pathlib import Path
from typing import Union

from waffle_utils.file import io

from waffle_hub import EXPORT_MAP
from waffle_hub.dataset import Dataset
from waffle_hub.hub.model.base_model import Model
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.hub.train.base_trainer import Trainer
from waffle_hub.schema.configs import ModelConfig
from waffle_hub.schema.fields.category import Category
from waffle_hub.type import TaskType


class BaseManager(Model, Trainer):
    """
    Base Training Manager
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
        root_dir: Path,
        name: str,
        task: Union[str, TaskType],
        model_type: str,
        model_size: str,
        categories: list[Union[str, int, float, dict, Category]],
        load: bool = False,
    ):
        Model.__init__(
            self,
            root_dir=root_dir,
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
            load=load,
        )
        Trainer.__init__(
            self,
            root_dir=root_dir,
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

        self.backend = self.BACKEND_NAME

    # Model abstract method
    @abstractmethod
    def get_model(self) -> ModelWrapper:
        raise NotImplementedError

    @abstractmethod
    def _get_preprocess(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_postprocess(self, *args, **kwargs):
        raise NotImplementedError

    # Trainer abstract method
    @abstractmethod
    def training(self):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError

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
    ) -> (Path, Path):
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
