import os
import warnings
from pathlib import Path
from typing import Union

import autocare_dlt
import tbparse
import torch
from autocare_dlt.core.model import build_model
from autocare_dlt.tools import train
from box import Box
from torch import nn
from torchvision import transforms as T
from waffle_utils.callback import BaseCallback
from waffle_utils.file import io

from waffle_hub.hub.manager.base_manager import BaseManager
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.fields.category import Category
from waffle_hub.schema.state import TrainState
from waffle_hub.type import BackendType, DataType, TaskType

from .autocare_dlt_callback import AutocareDLTCallback
from .config import DATA_TYPE_MAP, DEFAULT_PARAMS, MODEL_TYPES, WEIGHT_PATH


class AutocareDltManager(BaseManager):
    """
    Autocare dlt Manager
    """

    BACKEND_NAME = str(BackendType.AUTOCARE_DLT.value)
    BACKEND_VERSION = "autocare-dlt"
    MULTI_GPU_TRAIN = False
    MODEL_TYPES = MODEL_TYPES
    DEFAULT_PARAMS = DEFAULT_PARAMS
    DEFAULT_ADVANCE_PARAMS = {}

    DATA_TYPE_MAP = DATA_TYPE_MAP
    WEIGHT_PATH = WEIGHT_PATH

    def __init__(
        self,
        root_dir: Path,
        name: str,
        task: Union[str, TaskType],
        model_type: str,
        model_size: str,
        categories: list[Union[str, int, float, dict, Category]],
        callbacks: list[BaseCallback] = None,
        load: bool = False,
        train_state: TrainState = None,
    ):
        autocare_dlt_callbacks = [AutocareDLTCallback()]
        if callbacks is not None:
            autocare_dlt_callbacks.extend(callbacks)

        super().__init__(
            root_dir=root_dir,
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
            callbacks=autocare_dlt_callbacks,
            load=load,
            train_state=train_state,
        )

        if self.BACKEND_VERSION is not None and autocare_dlt.__version__ != self.BACKEND_VERSION:
            warnings.warn(
                f"You've loaded the Hub created with autocare_dlt=={self.BACKEND_VERSION}, \n"
                + f"but the installed version is {autocare_dlt.__version__}."
            )

        self.model_json_output_path = self.root_dir / self.CONFIG_DIR / "model.json"

    def get_model(self) -> ModelWrapper:
        """Get model.
        Returns:
            ModelWrapper: Model wrapper
        """
        self.check_train_sanity()

        # get adapt functions
        preprocess = self._get_preprocess()
        postprocess = self._get_postprocess()

        # get model
        categories = [x["name"] for x in self.get_categories()]
        cfg = io.load_json(self.model_json_output_path)
        cfg["ckpt"] = str(self.best_ckpt_file)
        if self.task == TaskType.TEXT_RECOGNITION:
            cfg["model"]["Prediction"]["num_classes"] = len(categories) + 1
        else:
            cfg["model"]["head"]["num_classes"] = len(categories)
        cfg["num_classes"] = len(categories)
        model, categories = build_model(Box(cfg), strict=True)

        # return model wrapper
        return ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
            task=self.task,
            categories=self.categories,
        )

    def _get_preprocess(self, *args, **kwargs):

        if self.task == TaskType.OBJECT_DETECTION:
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x, *args, **kwargs):
                return normalize(x)

        elif self.task == TaskType.CLASSIFICATION:
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x, *args, **kwargs):
                return normalize(x)

        elif self.task == TaskType.TEXT_RECOGNITION:
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x, *args, **kwargs):
                return normalize(x)

        elif self.task == TaskType.SEMANTIC_SEGMENTATION:
            normalize = T.Normalize([0], [1], inplace=True)
            gray_sacle = T.Grayscale()

            def preprocess(x, *args, **kwargs):
                return normalize(gray_sacle(x))

        else:
            raise NotImplementedError(f"task {self.task} is not supported yet")

        return preprocess

    def _get_postprocess(self, *args, **kwargs):

        if self.task == TaskType.OBJECT_DETECTION:

            def inner(x: torch.Tensor, *args, **kwargs):
                xyxy = x[0]
                scores = x[1]
                class_ids = x[2]

                return xyxy, scores, class_ids

        elif self.task == TaskType.CLASSIFICATION:

            def inner(x: torch.Tensor, *args, **kwargs):
                x = [t.squeeze(-1).squeeze(-1) for t in x]
                return x

        elif self.task == TaskType.TEXT_RECOGNITION:

            def inner(x: torch.Tensor, *args, **kwargs):
                scores, character_class_ids = x.max(dim=-1)
                return character_class_ids, scores

        elif self.task == TaskType.SEMANTIC_SEGMENTATION:

            def inner(x: torch.Tensor, *args, **kwargs):
                return x

        else:
            raise NotImplementedError(f"task {self.task} is not supported yet")

        return inner

    # Trainer
    def get_metrics(self) -> list[dict]:
        """Get metrics from tensorboard log directory.
        Args:
            tbdir (Union[str, Path]): tensorboard log directory
        Returns:
            list[dict]: list of metrics
        """
        tb_log_dir = self.artifacts_dir / "train" / "tensorboard"
        if not tb_log_dir.exists():
            return []

        sr = tbparse.SummaryReader(tb_log_dir)
        df = sr.scalars

        # Sort the data frame by step.
        # Make a list of dictionaries of tag and value.
        if df.empty:
            return []

        metrics = (
            df.sort_values("step")
            .groupby("step")
            .apply(lambda x: [{"tag": s, "value": v} for s, v in zip(x.tag, x.value)])
            .to_list()
        )

        return metrics
