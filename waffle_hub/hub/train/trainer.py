import os
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path, PurePath
from typing import Union

import torch

from waffle_hub.dataset import Dataset
from waffle_hub.hub.eval.evaluator import Evaluator
from waffle_hub.schema.configs import TrainConfig
from waffle_hub.schema.result import TrainResult
from waffle_hub.schema.running_status import TrainingStatus
from waffle_hub.utils.memory import device_context
from waffle_hub.utils.metric_logger import MetricLogger


class BaseTrainHooks(ABC):
    def __init__(self):
        pass

    # hooks
    def before_train(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass

    def on_train_start(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass

    def training(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass

    def on_train_end(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass

    def after_train(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass


class DefaultTrainHooks(BaseTrainHooks):
    def __init__(self, backend: str, artifacts_dir: Path, multi_gpu_train: bool):
        super().__init__()
        self.backend = backend
        self.artifacts_dir = artifacts_dir
        self.multi_gpu_train = multi_gpu_train

    # hooks
    def before_train(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        # check device
        device = self.device
        if device == "cpu":
            # logger.info("CPU training")
            pass
        elif device.isdigit():
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available.")
        elif "," in device:
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available.")
            if not self.multi_gpu_train:
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

    def on_train_start(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass

    def training(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass

    def on_train_end(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass

    def after_train(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass


class BaseTrainer:
    # Train spec, abstract property
    MULTI_GPU_TRAIN = None  ## ??
    DEFAULT_PARAMS = None
    DEFAULT_ADVANCE_PARAMS = {}

    # directory settting
    ARTIFACTS_DIR = Path("artifacts")
    TRAIN_LOG_DIR = Path("logs")
    WEIGHTS_DIR = Path("weights")

    # train results file name
    LAST_CKPT_FILE = "last_ckpt.pt"
    BEST_CKPT_FILE = "best_ckpt.pt"  # TODO: best metric?
    METRIC_FILE = "metrics.json"

    def __init__(self, root_dir: Path):
        # abstract property
        if self.MULTI_GPU_TRAIN is None:
            raise AttributeError("MULTI_GPU_TRAIN must be specified.")

        if self.DEFAULT_PARAMS is None:
            raise AttributeError("DEFAULT_PARAMS must be specified.")

        self.root_dir = Path(root_dir)
        # hooks
        default_hooks = DefaultTrainHooks(
            backend=self.backend,
            artifacts_dir=self.artifacts_dir,
            multi_gpu_train=self.MULTI_GPU_TRAIN,
        )
        self.hook_classes = OrderedDict({0: default_hooks})
        self.hooks_idx = 1

    # path properties
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

    # abstract method
    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def _init_hooks(self):
        """Init hooks each backend."""
        raise NotImplementedError

    # common
    def train(
        self,
        dataset_path: Union[str, Path],
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
            dataset_path (Union[str, Path]): dataset path
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
            rank = os.getenv("RANK", -1)
            if self.artifacts_dir.exists() and rank in [
                -1,
                0,
            ]:  # TODO: need to ensure that training is not already running
                raise FileExistsError(
                    f"{self.artifacts_dir}\n"
                    "Train artifacts already exist. Remove artifact to re-train [delete_artifact]."
                )

            # define status, result
            status = TrainingStatus(save_path=self.training_status_file)
            metric_logger = MetricLogger(
                name=self.name,
                log_dir=self.train_log_dir,
                func=self.get_metrics,
                interval=10,
                prefix="waffle",
                status=status,
            )
            result = TrainResult()

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
                cfg.image_size
                if isinstance(cfg.image_size, list)
                else [cfg.image_size, cfg.image_size]
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

            cfg.last_ckpt_file = self.last_ckpt_file
            cfg.best_ckpt_file = self.best_ckpt_file

            # train run
            # set status
            status.set_running()
            status.set_total_step(cfg.epochs)
            metric_logger.start()

            self.run_hooks("before_train", cfg, status, result)
            self.run_hooks("on_train_start", cfg, status, result)
            self.run_hooks("training", cfg, status, result)
            self.run_hooks("on_train_end", cfg, status, result)
            self.run_hooks("after_train", cfg, status, result)

            # write result
            result.best_ckpt_file = cfg.best_ckpt_file
            result.last_ckpt_file = cfg.last_ckpt_file
            result.metrics = self.get_metrics()

            status.set_success()
            metric_logger.stop()
        except FileExistsError as e:
            status.set_failed(e)
            metric_logger.stop()
            raise e
        except (KeyboardInterrupt, SystemExit) as e:
            status.set_stopped(e)
            metric_logger.stop()
            if self.artifact_dir.exists():
                self.on_train_end(cfg)
            raise e
        except Exception as e:
            status.set_failed(e)
            metric_logger.stop()
            if self.artifact_dir.exists():
                io.remove_directory(self.artifact_dir)
            raise e

        # TODO: evaluate

        # result.eval_metrics = self.get_evaluate_result()

        return result

    # hooks
    def run_hooks(self, event: str, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        for cls_id, hook_cls in self.hook_classes.items():
            method = getattr(hook_cls, event, None)
            if method is None:
                continue
            if not callable(method):
                warnings.warn(
                    f"Skipping the hook {hook_cls.__class__.__name__}, becuase it is not callable."
                )
                continue
            method(cfg, status, result)

    def register_hook(self, hook_cls: BaseTrainHooks):
        if not isinstance(hook_cls, BaseTrainHooks):
            raise TypeError(f"hook_cls must be subclass of BaseTrainHooks, not {type(hook_cls)}")
        self.hook_classes[self.hooks_idx] = hook_cls
        self.hooks_idx += 1

    def delete_hook(self, cls_id: int):
        if cls_id == 0:
            raise ValueError("Default hooks cannot be deleted.")
        self.hook_classes.pop(cls_id)

    def get_hooks(self) -> list[tuple[int, str]]:
        return [
            (cls_id, hook_cls.__class__.__name__) for cls_id, hook_cls in self.hook_classes.items()
        ]
