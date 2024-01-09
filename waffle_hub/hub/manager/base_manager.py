import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path, PurePath
from typing import Union

import torch
from waffle_utils.callback import BaseCallback
from waffle_utils.file import io
from waffle_utils.validator import setter_type_validator

from waffle_hub import EXPORT_MAP, TrainStatus
from waffle_hub.dataset import Dataset
from waffle_hub.hub.evaluator.evaluator import Evaluator
from waffle_hub.hub.manager.hook import BaseTrainHook
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import ModelConfig, TrainConfig
from waffle_hub.schema.fields.category import Category
from waffle_hub.schema.result import TrainResult
from waffle_hub.schema.state import TrainState
from waffle_hub.type import TaskType


class BaseManager(BaseTrainHook, ABC):
    """
    Base Manager
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
    CONFIG_DIR = Path("configs")
    ARTIFACTS_DIR = Path("artifacts")
    TRAIN_LOG_DIR = Path("logs")
    WEIGHTS_DIR = Path("weights")

    # train config file name
    MODEL_CONFIG_FILE = "model.yaml"
    TRAIN_CONFIG_FILE = "train.yaml"

    # train results file name
    LAST_CKPT_FILE = "last_ckpt.pt"
    BEST_CKPT_FILE = "best_ckpt.pt"  # TODO: best metric?
    METRIC_FILE = "metrics.json"

    def __init__(
        self,
        root_dir: Path,
        name: str,
        task: Union[str, TaskType] = None,
        model_type: str = None,
        model_size: str = None,
        categories: list[Union[str, int, float, dict, Category]] = None,
        callbacks: list[BaseCallback] = None,
        load: bool = False,
        train_state: TrainState = None,
    ):
        # abstract property
        if self.BACKEND_NAME is None:
            raise AttributeError("BACKEND_NAME must be specified.")

        if self.BACKEND_VERSION is None:
            raise AttributeError("BACKEND_VERSION must be specified.")

        if self.MODEL_TYPES is None:
            raise AttributeError("MODEL_TYPES must be specified.")

        if self.DEFAULT_PARAMS is None:
            raise AttributeError("DEFAULT_PARAMS must be specified.")

        if self.MULTI_GPU_TRAIN is None:
            raise AttributeError("MULTI_GPU_TRAIN must be specified.")

        super().__init__(callbacks=callbacks)
        self.root_dir = Path(root_dir)
        self.name = name
        self.task = TaskType.from_str(task).value if task else self.get_available_tasks()[0]
        self.model_type = model_type if model_type else self.get_available_model_types(self.task)[0]
        self.model_size = (
            model_size
            if model_size
            else self.get_available_model_sizes(self.task, self.model_type)[0]
        )
        self.categories = categories
        self.backend = self.BACKEND_NAME

        if load:
            self.state = (
                train_state if train_state is not None else TrainState(status=TrainStatus.INIT)
            )
        else:
            if self.model_config_file.exists():
                raise FileExistsError("Manager already exists. Try to 'load' function.")
            self.state = TrainState(status=TrainStatus.INIT)

        self.result = TrainResult()

        self.save_model_config(
            model_config_file=self.model_config_file,
        )

    # properties
    @property
    def task(self) -> str:
        """Task Name"""
        return self.__task

    @task.setter
    @setter_type_validator(Union[str, TaskType])
    def task(self, v):
        if v not in list(self.MODEL_TYPES.keys()):
            raise ValueError(
                f"Task {v} is not supported. Choose one of {[task.value for task in list(self.MODEL_TYPES.keys())]}"
            )
        self.__task = str(v.value) if isinstance(v, TaskType) else str(v).lower()

    @property
    def model_type(self) -> str:
        """Model Type"""
        return self.__model_type

    @model_type.setter
    @setter_type_validator(str)
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
    @setter_type_validator(str)
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
    @setter_type_validator(str)
    def version(self, v):
        self.__version = v

    @property
    def categories(self) -> list[Category]:
        return self.__categories

    @categories.setter
    @setter_type_validator(list)
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

    # path properties
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

    @property
    def artifacts_dir(self) -> Path:
        """Artifacts Directory. This is raw output of each backend."""
        return self.root_dir / self.ARTIFACTS_DIR

    @property
    def train_log_dir(self) -> Path:
        """Train Log Directory."""
        return self.root_dir / self.TRAIN_LOG_DIR

    @property
    def weights_dir(self) -> Path:
        """Weights Directory."""
        return self.root_dir / self.WEIGHTS_DIR

    @property
    def last_ckpt_file(self) -> Path:
        return self.weights_dir / self.LAST_CKPT_FILE

    @property
    def best_ckpt_file(self) -> Path:
        return self.weights_dir / self.BEST_CKPT_FILE

    @property
    def metric_file(self) -> Path:
        return self.root_dir / self.METRIC_FILE

    # manager common methods
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
            version=self.BACKEND_VERSION,
            task=self.task,
            model_type=self.model_type,
            model_size=self.model_size,
            categories=list(map(lambda x: x.to_dict(), self.categories)),
        ).save_yaml(self.model_config_file)

    def get_categories(self) -> list[Category]:
        return self.categories

    def get_category_names(self) -> list[str]:
        return [category.name for category in self.categories]

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
        cls,
        root_dir: str,
        name: str,
        model_config_file_path: Union[str, Path],
        callbacks: list[BaseCallback] = None,
    ):
        """
        Create Manager from model config file

        Args:
            root_dir (str): Root directory
            name (str): Model name
            model_config_file_path (Union[str, Path]): Model config file path
            callbacks (list[BaseCallback], optional): Callbacks. Defaults to None.

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
            callbacks=callbacks,
        )

    @classmethod
    def load(
        cls, root_dir: str, train_state: TrainState = None, callbacks: list[BaseCallback] = None
    ):
        """
        Load Manager from model config file in root directory

        Args:
            root_dir (str): Root directory
            train_state (TrainState, optional): Train state. Defaults to None.
            callbacks (list[BaseCallback], optional): Callbacks. Defaults to None.

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
            callbacks=callbacks,
            load=True,
            train_state=train_state,
        )

    @classmethod
    def get_available_tasks(cls) -> list[str]:
        """
        Get available tasks

        Returns:
            list[str]: Available tasks
        """
        return list(cls.MODEL_TYPES.keys())

    @classmethod
    def get_available_model_types(cls, task: str) -> list[str]:
        """
        Get available model types

        Args:
            task (str): Task name

        Raises:
            ModuleNotFoundError: If backend is not supported

        Returns:
            list[str]: Available model types
        """
        if task not in list(cls.MODEL_TYPES.keys()):
            raise ValueError(f"{task} is not supported with {cls.BACKEND_NAME}")
        task = TaskType.from_str(task).value
        return list(cls.MODEL_TYPES[task].keys())

    @classmethod
    def get_available_model_sizes(cls, task: str, model_type: str) -> list[str]:
        """
        Get available model sizes

        Args:
            task (str): Task name
            model_type (str): Model type

        Raises:
            ValueError: If backend is not supported

        Returns:
            list[str]: Available model sizes
        """
        if task not in list(cls.MODEL_TYPES.keys()):
            raise ValueError(f"{task} is not supported with {cls.BACKEND_NAME}")
        task = TaskType.from_str(task).value
        if model_type not in list(cls.MODEL_TYPES[task].keys()):
            raise ValueError(f"{model_type} is not supported with {cls.BACKEND_NAME}")
        model_sizes = cls.MODEL_TYPES[task][model_type]
        return model_sizes if isinstance(model_sizes, list) else list(model_sizes.keys())

    @classmethod
    def get_default_params(cls, task: str, model_type: str, model_size: str) -> dict:
        """
        Get default params

        Args:
            task (str): Task name
            model_type (str): Model type
            model_size (str): Model size

        Raises:
            ValueError: If backend is not supported

        Returns:
            dict: Default params
        """
        if task not in list(cls.MODEL_TYPES.keys()):
            raise ValueError(f"{task} is not supported with {cls.BACKEND_NAME}")
        task = TaskType.from_str(task).value
        if model_type not in cls.MODEL_TYPES[task]:
            raise ValueError(f"{model_type} is not supported with {cls.BACKEND_NAME}")
        if model_size not in cls.MODEL_TYPES[task][model_type]:
            raise ValueError(f"{model_size} is not supported with {cls.BACKEND_NAME}")
        return cls.DEFAULT_PARAMS[task][model_type][model_size]

    def delete_manager(self):
        """
        Delete manager.
        """
        if self.root_dir.exists():
            io.remove_directory(self.root_dir, recursive=True)
        del self
        return None

    def delete_artifacts(self):
        """
        Delete manager.
        """
        if self.train_config_file.exists():
            io.remove_file(self.train_config_file)
        if self.artifacts_dir.exists():
            io.remove_directory(self.artifacts_dir, recursive=True)
        if self.weights_dir.exists():
            io.remove_directory(self.weights_dir, recursive=True)
        if self.train_log_dir.exists():
            io.remove_directory(self.train_log_dir, recursive=True)
        if self.metric_file.exists():
            io.remove_file(self.metric_file)

        self.state = TrainState(status=TrainStatus.INIT)
        return None

    # Configs methods
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
            version=self.BACKEND_VERSION,
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

    # Train abstract method
    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError

    def check_train_sanity(self) -> bool:
        """Check if all essential files are exist.

        Returns:
            bool: True if all files are exist else False
        """
        if not self.best_ckpt_file.exists():  # last_ckpt_file.exists() or ??
            raise ValueError("Train first! hub.train(...).")
        return True

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
        try:
            # check if it is already trained
            self.run_default_hook("setup")
            self.run_callback_hooks("setup", self)

            # check if it is already trained # TODO: resume
            rank = os.getenv("RANK", -1)
            if self.artifacts_dir.exists() and rank in [
                -1,
                0,
            ]:  # TODO: need to ensure that training is not already running
                raise FileExistsError(
                    f"{self.artifacts_dir}\n"
                    "Train artifacts already exist. Remove artifact to re-train [delete_artifacts]."
                )

            # parse dataset
            export_path, dataset_path = self._parse_dataset(dataset, dataset_root_dir)

            self.cfg = self._parse_train_config(
                dataset_path=export_path,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                learning_rate=learning_rate,
                letter_box=letter_box,
                pretrained_model=pretrained_model,
                device=device,
                workers=workers,
                seed=seed,
                advance_params=advance_params,
                verbose=verbose,
            )

            # save train config
            self.save_train_config(self.cfg, self.train_config_file)

            # check device
            device = self.cfg.device
            if device == "cpu":
                # logger.info("CPU training")
                pass
            elif device.isdigit():
                if not torch.cuda.is_available():
                    raise ValueError("CUDA is not available.")
            elif "," in device:
                if not torch.cuda.is_available():
                    raise ValueError("CUDA is not available.")
                if not self.MULTI_GPU_TRAIN:
                    raise ValueError(f"{self.backend} does not support MULTI_GPU_TRAIN.")
                if len(device.split(",")) > torch.cuda.device_count():
                    raise ValueError(
                        f"GPU number is not enough. {device}\n"
                        + f"Given device: {device}\n"
                        + f"Available device count: {torch.cuda.device_count()}"
                    )
                # TODO: occurs unexpected errors
                # if not all([int(x) < torch.cuda.device_count() for x in device.split(",")]):
                #     raise IndexError(
                #         f"GPU index is out of range. device id should be smaller than {torch.cuda.device_count()}\n"
                #     )
                # logger.info(f"Multi GPU training: {device}")
            else:
                raise ValueError(f"Invalid device: {device}\n" + "Please use 'cpu', '0', '0,1,2,3'")

            self.run_default_hook("before_train")
            self.run_callback_hooks("before_train", self)

            self._train()

            self._evaluate(dataset_path)

            self.run_default_hook("after_train")
            self.run_callback_hooks("after_train", self)

        except FileExistsError as e:
            raise e
        except (KeyboardInterrupt, SystemExit) as e:
            self.run_default_hook("on_exception_stopped", e)
            self.run_callback_hooks("on_exception_stopped", self, e)
            if self.artifacts_dir.exists():
                self.run_default_hook("on_train_end")
                self.run_callback_hooks("on_train_end", self)
            raise e
        except Exception as e:
            self.run_default_hook("on_exception_failed", e)
            self.run_callback_hooks("on_exception_failed", self, e)
            if self.artifacts_dir.exists():
                io.remove_directory(self.artifacts_dir, recursive=True)
            raise e
        finally:
            self.run_default_hook("teardown")
            self.run_callback_hooks("teardown", self)

        return self.result

    def _train(self):
        self.run_default_hook("on_train_start")
        self.run_callback_hooks("on_train_start", self)

        self.run_default_hook("training")
        self.run_callback_hooks("training", self)

        self.run_default_hook("on_train_end")
        self.run_callback_hooks("on_train_end", self)

    def _parse_dataset(
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

        # check task match
        if dataset.task.lower() != self.task.lower():
            raise ValueError(
                f"Dataset task is not matched with hub task. Dataset task: {dataset.task}, Hub task: {self.task}"
            )

        # check category match
        if not self.categories:
            self.categories = dataset.get_categories()
            self.save_model_config(self.model_config_file)
        elif set(dataset.get_category_names()) != set(self.get_category_names()):
            raise ValueError(
                "Dataset categories are not matched with hub categories. \n"
                + f"Dataset categories: {dataset.get_category_names()}, Hub categories: {self.get_category_names()}"
            )

        # convert dataset to backend format if not exist
        export_dir = dataset.export_dir / EXPORT_MAP[self.backend]
        if not export_dir.exists():
            ## LOGGER!!
            # logger.info(f"[Dataset] Exporting dataset to {self.backend} format...")
            export_dir = dataset.export(self.backend)
            # logger.info("[Dataset] Exporting done.")

        return export_dir, dataset.dataset_dir

    def _parse_train_config(
        self,
        dataset_path: Path,
        epochs: int,
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
    ) -> TrainConfig:
        # parse train config
        cfg = TrainConfig(
            dataset_path=dataset_path,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate,
            letter_box=letter_box,
            pretrained_model=pretrained_model,
            device=device,
            workers=workers,
            seed=seed,
            advance_params=advance_params if advance_params else {},
            verbose=verbose,
        )

        ## overwrite train config with default config
        for k, v in cfg.to_dict().items():
            if v is None:
                field_value = getattr(
                    self.DEFAULT_PARAMS[self.task][self.model_type][self.model_size], k
                )
                setattr(cfg, k, field_value)
        cfg.image_size = (
            cfg.image_size if isinstance(cfg.image_size, list) else [cfg.image_size, cfg.image_size]
        )

        ## overwrite train advance config
        if cfg.advance_params:
            if isinstance(cfg.advance_params, (str, PurePath)):
                # check if it is yaml or json
                if Path(cfg.advance_params).exists():
                    if Path(cfg.advance_params).suffix in [".yaml", ".yml"]:
                        cfg.advance_params = io.load_yaml(cfg.advance_params)
                    elif Path(cfg.advance_params).suffix in [".json"]:
                        cfg.advance_params = io.load_json(cfg.advance_params)
                    else:
                        raise ValueError(
                            f"Advance parameter file should be yaml or json. {cfg.advance_params}"
                        )
                else:
                    raise FileNotFoundError(f"Advance parameter file is not exist.")
            elif not isinstance(cfg.advance_params, dict):
                raise ValueError(
                    f"Advance parameter should be dictionary or file path. {cfg.advance_params}"
                )

            default_advance_param = self.get_default_advance_train_params()
            for key in cfg.advance_params.keys():
                if key not in default_advance_param:
                    raise ValueError(
                        f"Advance parameter {key} is not supported.\n"
                        + f"Supported parameters: {list(default_advance_param.keys())}"
                    )
        return cfg

    def _evaluate(self, dataset_path: Path):
        # evaluate
        self.evaluator = Evaluator(root_dir=self.root_dir, model=self.get_model())

        self.run_default_hook("on_evaluate_start")
        self.run_callback_hooks("on_evaluate_start", self)

        result = self.evaluator.evaluate(
            dataset=dataset_path,
            batch_size=self.cfg.batch_size,
            image_size=self.cfg.image_size,
            letter_box=self.cfg.letter_box,
            device=self.cfg.device,
            workers=self.cfg.workers,
        )
        self.result.eval_metrics = result.eval_metrics

        self.run_default_hook("on_evaluate_end")
        self.run_callback_hooks("on_evaluate_end", self)
