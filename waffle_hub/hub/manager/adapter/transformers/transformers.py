import os
import warnings
from pathlib import Path
from typing import Callable, Union

import torch
import transformers
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
)
from transformers.utils import ModelOutput
from waffle_utils.callback import BaseCallback
from waffle_utils.file import io

from waffle_hub.hub.manager.adapter.transformers.transformers_callback import (
    TransformersTrainCallback,
)
from waffle_hub.hub.manager.base_manager import BaseManager
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.fields.category import Category
from waffle_hub.schema.state import TrainState
from waffle_hub.type import BackendType, TaskType

from .config import DEFAULT_PARAMS, MODEL_TYPES


class TransformersManager(BaseManager):
    """
    Transformer Manager
    """

    BACKEND_NAME = str(BackendType.TRANSFORMERS.value)
    BACKEND_VERSION = "4.34.1"
    MODEL_TYPES = MODEL_TYPES
    MULTI_GPU_TRAIN = False
    DEFAULT_PARAMS = DEFAULT_PARAMS
    DEFAULT_ADVANCE_PARAMS = {}

    # Override
    LAST_CKPT_FILE = "last_ckpt"
    BEST_CKPT_FILE = "best_ckpt"

    def __init__(
        self,
        root_dir: Path,
        name: str,
        task: Union[str, TaskType],
        model_type: str,
        model_size: str,
        categories: list[Union[str, int, float, dict, Category]] = None,
        callbacks: list[BaseCallback] = None,
        load: bool = False,
        train_state: TrainState = None,
    ):
        transformers_callbacks = [TransformersTrainCallback()]
        if callbacks is not None:
            transformers_callbacks.extend(callbacks)

        super().__init__(
            root_dir=root_dir,
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
            callbacks=transformers_callbacks,
            load=load,
            train_state=train_state,
        )

        if self.BACKEND_VERSION is not None and transformers.__version__ != self.BACKEND_VERSION:
            warnings.warn(
                f"You've loaded the Hub created with transformers=={self.BACKEND_VERSION}, \n"
                + f"but the installed version is {transformers.__version__}."
            )

    # override
    @property
    def last_ckpt_file(self) -> Path:
        return self.weights_dir / TransformersManager.LAST_CKPT_FILE

    @property
    def best_ckpt_file(self) -> Path:
        return self.weights_dir / TransformersManager.BEST_CKPT_FILE

    # Model
    def get_model(self) -> ModelWrapper:
        self.check_train_sanity()

        # get adapt functions
        preprocess = self._get_preprocess()
        postprocess = self._get_postprocess()

        # get model
        if self.task == TaskType.OBJECT_DETECTION:
            model = AutoModelForObjectDetection.from_pretrained(
                str(self.best_ckpt_file),
            )
        elif self.task == TaskType.CLASSIFICATION:
            model = AutoModelForImageClassification.from_pretrained(
                str(self.best_ckpt_file),
            )
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented.")

        model = ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
            task=self.task,
            categories=self.categories,
        )

        return model

    def _get_preprocess(self, pretrained_model: str = None) -> Callable:
        if pretrained_model is None:
            pretrained_model = self.best_ckpt_file
        image_processer = AutoImageProcessor.from_pretrained(pretrained_model)

        normalize = T.Normalize(image_processer.image_mean, image_processer.image_std, inplace=True)

        def preprocess(x, *args, **kwargs):
            return normalize(x)

        return preprocess

    def _get_postprocess(self: str, pretrained_model: str = None) -> Callable:
        if pretrained_model is None:
            pretrained_model = self.best_ckpt_file
        image_processer = AutoImageProcessor.from_pretrained(pretrained_model)

        if self.task == TaskType.CLASSIFICATION:

            def inner(x: ModelOutput, *args, **kwargs) -> torch.Tensor:
                return [x.logits]

        elif self.task == TaskType.OBJECT_DETECTION:
            post_process = image_processer.post_process_object_detection

            def inner(x: ModelOutput, *args, **kwargs) -> torch.Tensor:

                x = post_process(x, threshold=-1)

                xyxy = list(map(lambda x: x["boxes"], x))
                confidences = list(map(lambda x: x["scores"], x))
                category_ids = list(map(lambda x: x["labels"], x))

                return xyxy, confidences, category_ids

        else:
            raise NotImplementedError(f"Task {self.task} is not implemented.")

        return inner

    # Trainer
    def get_metrics(self) -> list[list[dict]]:
        return io.load_json(self.metric_file) if self.metric_file.exists() else []
