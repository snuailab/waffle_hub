from abc import abstractmethod
from pathlib import Path
from typing import Union

from waffle_dough.type.task_type import TaskType
from waffle_hub import EXPORT_MAP
from waffle_hub.dataset import Dataset
from waffle_hub.hub.model.base_model import Model
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.hub.train.base_trainer import Trainer
from waffle_hub.schema.fields.category import Category


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
    ):
        Model.__init__(
            self,
            root_dir=root_dir,
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
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
    def parse_dataset(
        self, dataset: Union[Dataset, str, Path], dataset_root_dir: str = None
    ) -> Path:
        # parse dataset
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
        if dataset.task.upper() != self.task.upper():
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

        return export_dir
