import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path, PurePath
from typing import Union

import torch
from waffle_utils.file import io

from waffle_hub import TrainStatus
from waffle_hub.dataset import Dataset
from waffle_hub.hub.eval.evaluator import Evaluator
from waffle_hub.hub.train.hook import BaseTrainHook
from waffle_hub.schema.configs import TrainConfig
from waffle_hub.schema.result import TrainResult
from waffle_hub.schema.state import TrainState
from waffle_hub.utils.memory import device_context
from waffle_hub.utils.metric_logger import MetricLogger


class BaseTrainer(BaseTrainHook, ABC):
    # Train spec, abstract property
    MULTI_GPU_TRAIN = None

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

        self.root_dir = Path(root_dir)
        self.state = TrainState(status=TrainStatus.INIT)
        self.result = TrainResult()

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

    # common
    @device_context
    def train(
        self,
        train_cfg: TrainConfig,
    ) -> TrainResult:
        """Start Train

        Args:
            train_cfg (TrainConfig): train config

        Returns:
            TrainResult: train result
        """
        try:
            # check if it is already trained
            self.run_default_hook("setup")
            self.run_callback_hooks("setup", self)

            rank = os.getenv("RANK", -1)
            if self.artifacts_dir.exists() and rank in [
                -1,
                0,
            ]:  # TODO: need to ensure that training is not already running
                raise FileExistsError(
                    f"{self.artifacts_dir}\n"
                    "Train artifacts already exist. Remove artifact to re-train [delete_artifact]."
                )

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

            self.cfg = train_cfg
            self.run_default_hook("before_train")
            self.run_callback_hooks("before_train", self)

            self._train()

            self.run_default_hook("after_train")
            self.run_callback_hooks("after_train", self)

            # write result
            self.result.best_ckpt_file = self.best_ckpt_file
            self.result.last_ckpt_file = self.last_ckpt_file
            self.result.metrics = self.get_metrics()
        except FileExistsError as e:
            raise e
        except (KeyboardInterrupt, SystemExit) as e:
            self.run_default_hook("exception_stopped", e)
            self.run_callback_hooks("exception_stopped", self, e)
            if self.artifact_dir.exists():
                self.run_default_hook("on_train_end")
                self.run_callback_hooks("on_train_end", self)
            raise e
        except Exception as e:
            self.run_default_hook("exception_failed", e)
            self.run_callback_hooks("exception_failed", self, e)
            if self.artifact_dir.exists():
                io.remove_directory(self.artifact_dir)
            raise e
        finally:
            self.run_default_hook("teardown")
            self.run_callback_hooks("teardown", self)

        # TODO: evaluate

        # result.eval_metrics = self.get_evaluate_result()

        return self.result

    def _train(self):
        self.run_default_hook("on_train_start")
        self.run_callback_hooks("on_train_start", self)

        self.run_default_hook("training")
        self.run_callback_hooks("training", self)

        self.run_default_hook("on_train_end")
        self.run_callback_hooks("on_train_end", self)
