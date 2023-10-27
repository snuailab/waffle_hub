import os
import threading
import warnings
from abc import ABC, abstractmethod
from pathlib import Path, PurePath
from typing import Union

import torch
from waffle_utils.file import io

from waffle_hub.dataset import Dataset
from waffle_hub.schema.configs import TrainConfig
from waffle_hub.schema.result import TrainResult
from waffle_hub.utils.callback import TrainCallback
from waffle_hub.utils.memory import device_context
from waffle_hub.utils.metric_logger import MetricLogger


class Trainer(ABC):
    """
    Base class for for training manager

    args:
        root_dir (Path) : root directory
    """

    # Train spec, abstract property
    MULTI_GPU_TRAIN = None  ## ??
    DEFAULT_PARAMS = None
    DEFAULT_ADVANCE_PARAMS = {}

    # directory settting
    ARTIFACT_DIR = Path("artifacts")
    TRAIN_LOG_DIR = Path("logs")
    WEIGHTS_DIR = Path("weights")

    # train config file name
    TRAIN_CONFIG_FILE = "train.yaml"

    # train results file name
    LAST_CKPT_FILE = "last_ckpt.pt"
    BEST_CKPT_FILE = "best_ckpt.pt"  # TODO: best metric?
    METRIC_FILE = "metrics.json"

    def __init__(
        self,
        root_dir: Path,
    ):
        # abstract property
        if self.MULTI_GPU_TRAIN is None:
            raise AttributeError("MULTI_GPU_TRAIN must be specified.")

        if self.DEFAULT_PARAMS is None:
            raise AttributeError("DEFAULT_PARAMS must be specified.")

        self.root_dir = Path(root_dir)

    @property
    def artifact_dir(self) -> Path:
        """Artifact Directory. This is raw output of each backend."""
        return self.root_dir / self.ARTIFACT_DIR

    @property
    def weights_dir(self) -> Path:
        """Weights Directory."""
        return self.root_dir / self.WEIGHTS_DIR

    @property
    def train_log_dir(self) -> Path:
        """Train Log Directory."""
        return self.root_dir / self.TRAIN_LOG_DIR

    @property
    def train_config_file(self) -> Path:
        return self.config_dir / self.TRAIN_CONFIG_FILE

    @property
    def last_ckpt_file(self) -> Path:
        return self.weights_dir / self.LAST_CKPT_FILE

    @property
    def best_ckpt_file(self) -> Path:
        return self.weights_dir / self.BEST_CKPT_FILE

    @property
    def metric_file(self) -> Path:
        return self.root_dir / self.METRIC_FILE

    def train(
        self,
        dataset: Union[Dataset, str],
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
        hold: bool = True,
    ) -> TrainResult:
        """Start Train

        Args:
            dataset (Union[Dataset, str]): Waffle Dataset object or path or name.
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
            hold (bool, optional): hold process. Defaults to True.

        Returns:
            TrainResult: train result
        """

        @device_context("cpu" if device == "cpu" else device)
        def inner(callback: TrainCallback, result: TrainResult):
            try:
                metric_logger = MetricLogger(
                    name=self.name,
                    log_dir=self.train_log_dir,
                    func=self.get_metrics,
                    interval=10,
                    prefix="waffle",
                )
                metric_logger.start()
                self.before_train()
                self.on_train_start()
                self.save_train_config(self.train_cfg, self.train_config_file)
                self.training()
                self.on_train_end()
                # self.evaluate(
                #     dataset=dataset,
                #     batch_size=cfg.batch_size,
                #     image_size=cfg.image_size,
                #     letter_box=cfg.letter_box,
                #     device=cfg.device,
                #     workers=cfg.workers,
                # )
                self.after_train(result)
                metric_logger.stop()
                callback.force_finish()
            except FileExistsError as e:
                callback.force_finish()
                callback.set_failed()
                raise e
            except Exception as e:
                if self.artifact_dir.exists():
                    io.remove_directory(self.artifact_dir)
                callback.force_finish()
                callback.set_failed()
                raise e

        export_dir = self.parse_dataset(dataset, dataset_root_dir)

        # parse train config
        self.train_cfg = TrainConfig(
            dataset_path=export_dir,
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
        for k, v in self.train_cfg.to_dict().items():
            if v is None:
                field_value = getattr(
                    self.DEFAULT_PARAMS[self.task][self.model_type][self.model_size], k
                )
                setattr(self.train_cfg, k, field_value)
        self.train_cfg.image_size = (
            self.train_cfg.image_size
            if isinstance(self.train_cfg.image_size, list)
            else [self.train_cfg.image_size, self.train_cfg.image_size]
        )
        # overwrite train advance config
        if self.train_cfg.advance_params:
            if isinstance(self.train_cfg.advance_params, (str, PurePath)):
                # check if it is yaml or json
                if Path(self.train_cfg.advance_params).exists():
                    if Path(self.train_cfg.advance_params).suffix in [".yaml", ".yml"]:
                        self.train_cfg.advance_params = io.load_yaml(self.train_cfg.advance_params)
                    elif Path(self.train_cfg.advance_params).suffix in [".json"]:
                        self.train_cfg.advance_params = io.load_json(self.train_cfg.advance_params)
                    else:
                        raise ValueError(
                            f"Advance parameter file should be yaml or json. {self.train_cfg.advance_params}"
                        )
                else:
                    raise FileNotFoundError(f"Advance parameter file is not exist.")
            elif not isinstance(self.train_cfg.advance_params, dict):
                raise ValueError(
                    f"Advance parameter should be dictionary or file path. {self.train_cfg.advance_params}"
                )

            default_advance_param = self.get_default_advance_train_params()
            for key in self.train_cfg.advance_params.keys():
                if key not in default_advance_param:
                    raise ValueError(
                        f"Advance parameter {key} is not supported.\n"
                        + f"Supported parameters: {list(default_advance_param.keys())}"
                    )

        callback = TrainCallback(self.train_cfg.epochs + 1, self.get_metrics)
        result = TrainResult()
        result.callback = callback

        # TODO: hold arguemnt will be deprecated
        if hold:
            inner(callback, result)
        else:
            thread = threading.Thread(target=inner, args=(callback, result), daemon=True)
            callback.register_thread(thread)
            callback.start()

        return result

    # Train Hook
    def before_train(self):
        # check device
        device = self.train_cfg.device
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
            # if len(device.split(",")) > torch.cuda.device_count():  # TODO: torch.cuda.device_count() occurs unexpected errors
            #     raise ValueError(
            #         f"GPU number is not enough. {device}\n"
            #         + f"Given device: {device}\n"
            #         + f"Available device count: {torch.cuda.device_count()}"
            #     )
            # if not all([int(x) < torch.cuda.device_count() for x in device.split(",")]):
            #     raise IndexError(
            #         f"GPU index is out of range. device id should be smaller than {torch.cuda.device_count()}\n"
            #     )
            # logger.info(f"Multi GPU training: {device}")
        else:
            raise ValueError(f"Invalid device: {device}\n" + "Please use 'cpu', '0', '0,1,2,3'")

        # check if it is already trained
        rank = os.getenv("RANK", -1)
        if self.artifact_dir.exists() and rank in [
            -1,
            0,
        ]:  # TODO: need to ensure that training is not already running
            raise FileExistsError(
                f"{self.artifact_dir}\n"
                "Train artifacts already exist. Remove artifact to re-train (hub.delete_artifact())."
            )

    def on_train_start(self):
        pass

    @abstractmethod
    def training(self):
        raise NotImplementedError

    def on_train_end(self):
        pass

    def after_train(self, result: TrainResult):
        result.best_ckpt_file = self.best_ckpt_file
        result.last_ckpt_file = self.last_ckpt_file
        result.metrics = self.get_metrics()
        # result.eval_metrics = self.get_evaluate_result()

    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError

    def get_train_config(self) -> TrainConfig:
        """Get train config from train config yaml file.

        Returns:
            TrainConfig: train config
        """
        if not self.train_config_file.exists():
            warnings.warn("Train config file is not exist. Train first!")
            return None
        return TrainConfig.load(self.train_config_file)

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

    def load_train_config(self, train_config_path: Path):
        """Load train config from yaml file (set self.train_cfg from yaml file)

        Args:
            train_config_path (Path): file path for loading train config
        """
        if not train_config_path.exists():
            raise FileNotFoundError(f"{train_config_path} is not exist.")
        self.train_cfg = TrainConfig.load(train_config_path)

    def check_train_sanity(self) -> bool:
        """Check if all essential files are exist.

        Returns:
            bool: True if all files are exist else False
        """
        if not (
            self.model_config_file.exists()
            and self.best_ckpt_file.exists()
            # and self.last_ckpt_file.exists()
        ):
            raise FileNotFoundError("Train first! hub.train(...).")
        return True

    def delete_artifact(self):
        """Delete Artifact Directory. It can be trained again."""
        if self.artifact_dir.exists():
            io.remove_directory(self.artifact_dir)
        else:
            warnings.warn(f"{self.artifact_dir} is not exist.")  ## log
