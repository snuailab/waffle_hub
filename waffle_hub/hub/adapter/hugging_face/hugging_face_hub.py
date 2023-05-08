"""
Hugging Face Hub
See BaseHub documentation for more details about usage.
"""

from waffle_hub import get_installed_backend_version

BACKEND_NAME = "transformers"
BACKEND_VERSION = get_installed_backend_version(BACKEND_NAME)

import os
import warnings
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Callable, Union

import torch
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import ModelOutput
from waffle_utils.file import io

from datasets import load_from_disk

from waffle_hub import TaskType
from waffle_hub.hub.adapter.hugging_face.train_input_helper import (
    ClassifierInputHelper,
    ObjectDetectionInputHelper,
)
from waffle_hub.hub.base_hub import BaseHub
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import TrainConfig
from waffle_hub.utils.callback import TrainCallback

from .config import DEFAULT_PARAMAS, MODEL_TYPES


class CustomCallback(TrainerCallback):
    """
    This class is necessary to obtain logs for the training.
    """

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset,
                metric_key_prefix="train",
            )
            return control_copy


class HuggingFaceHub(BaseHub):

    # Override
    LAST_CKPT_FILE = "weights/last_ckpt"
    BEST_CKPT_FILE = "weights/best_ckpt"

    # Common
    MODEL_TYPES = MODEL_TYPES
    DEFAULT_PARAMAS = DEFAULT_PARAMAS

    def __init__(
        self,
        name: str,
        task: str,
        model_type: str,
        model_size: str,
        categories: Union[list[dict], list] = None,
        root_dir: str = None,
        backend: str = None,
        version: str = None,
    ):

        if backend is not None and backend != BACKEND_NAME:
            raise ValueError(f"you've loaded {backend}. backend must be {BACKEND_NAME}")

        if version is not None and version != BACKEND_VERSION:
            warnings.warn(
                f"you've loaded a {BACKEND_NAME}=={version} version while {BACKEND_NAME}=={BACKEND_VERSION} version is installed."
                "It will cause unexpected results."
            )

        super().__init__(
            name=name,
            backend=BACKEND_NAME,
            version=BACKEND_VERSION,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
            root_dir=root_dir,
        )

    # Override
    @cached_property
    def best_ckpt_file(self) -> Path:
        """Best Checkpoint File"""
        return self.hub_dir / HuggingFaceHub.BEST_CKPT_FILE

    @cached_property
    def last_ckpt_file(self) -> Path:
        """Last Checkpoint File"""
        return self.hub_dir / HuggingFaceHub.LAST_CKPT_FILE

    @classmethod
    def new(
        cls,
        name: str,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        categories: Union[list[dict], list] = None,
        root_dir: str = None,
    ):

        return cls(
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
            root_dir=root_dir,
        )

    def on_train_start(self, cfg: TrainConfig):
        # overwrite train config with default config
        cfg.pretrained_model = self.MODEL_TYPES[self.task][self.model_type][self.model_size]

        # setting
        if cfg.device != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        dataset = load_from_disk(cfg.dataset_path)

        if self.task == "classification":
            helper = ClassifierInputHelper(cfg.pretrained_model, cfg.image_size)
            cfg.train_input = helper.get_train_input()
            categories = dataset["train"].features["label"].names
            id2label = {index: x for index, x in enumerate(categories, start=0)}
            cfg.train_input.model = AutoModelForImageClassification.from_pretrained(
                cfg.pretrained_model,
                num_labels=len(id2label),
                ignore_mismatched_sizes=True,
            )

        elif self.task == "object_detection":
            helper = ObjectDetectionInputHelper(cfg.pretrained_model, cfg.image_size)
            cfg.train_input = helper.get_train_input()
            categories = dataset["train"].features["objects"].feature["category"].names
            id2label = {index: x for index, x in enumerate(categories, start=0)}
            label2id = {x: index for index, x in enumerate(categories, start=0)}
            cfg.train_input.model = AutoModelForObjectDetection.from_pretrained(
                cfg.pretrained_model,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            raise NotImplementedError

        transforms = helper.get_transforms()
        dataset["train"] = dataset["train"].with_transform(transforms)
        dataset["val"] = dataset["val"].with_transform(transforms)
        cfg.train_input.dataset = dataset

        cfg.train_input.training_args = TrainingArguments(
            output_dir=str(self.artifact_dir),
            per_device_train_batch_size=cfg.batch_size,
            num_train_epochs=cfg.epochs,
            remove_unused_columns=False,
            push_to_hub=False,
            logging_strategy="epoch" if cfg.verbose else "no",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=cfg.learning_rate,
            dataloader_num_workers=cfg.workers,
            seed=cfg.seed,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            load_best_model_at_end=False,
        )

    def training(self, cfg: TrainConfig, callback: TrainCallback):

        trainer = Trainer(
            model=cfg.train_input.model,
            args=cfg.train_input.training_args,
            data_collator=cfg.train_input.collator,
            train_dataset=cfg.train_input.dataset["train"],
            eval_dataset=cfg.train_input.dataset["val"],
            tokenizer=cfg.train_input.image_processor,
            compute_metrics=cfg.train_input.compute_metrics,
        )
        trainer.add_callback(CustomCallback(trainer))
        trainer.train()
        trainer.save_model(str(self.artifact_dir / "weights" / "last_ckpt"))
        trainer._load_best_model()
        trainer.save_model(str(self.artifact_dir / "weights" / "best_ckpt"))

        self.train_log = trainer.state.log_history

    def get_metrics(self) -> list[list[dict]]:
        metrics = []

        for epoch in range(0, len(self.train_log) - 1, 3):  # last is runtime info

            current_epoch_log = self.train_log[epoch]

            current_epoch_log.update(self.train_log[epoch + 1])
            current_epoch_log.update(self.train_log[epoch + 2])

            epoch_metrics = []
            for key, value in current_epoch_log.items():
                epoch_metrics.append({"tag": key, "value": value})
            metrics.append(epoch_metrics)

        return metrics

    def on_train_end(self, cfg: TrainConfig):
        io.copy_files_to_directory(
            self.artifact_dir / "weights" / "best_ckpt",
            self.best_ckpt_file,
            create_directory=True,
        )
        io.copy_files_to_directory(
            self.artifact_dir / "weights" / "last_ckpt",
            self.last_ckpt_file,
            create_directory=True,
        )
        io.save_json(self.get_metrics(), self.metric_file)

    def get_preprocess(self, pretrained_model: str = None) -> Callable:
        if pretrained_model is None:
            pretrained_model = self.best_ckpt_file
        image_processer = AutoImageProcessor.from_pretrained(pretrained_model)

        normalize = T.Normalize(image_processer.image_mean, image_processer.image_std, inplace=True)

        def preprocess(x, *args, **kwargs):
            return normalize(x)

        return preprocess

    def get_postprocess(self: str, pretrained_model: str = None) -> Callable:
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

    def get_model(self) -> ModelWrapper:
        self.check_train_sanity()

        # get adapt functions
        preprocess = self.get_preprocess()
        postprocess = self.get_postprocess()

        # get model
        if self.task == TaskType.OBJECT_DETECTION:
            model = AutoModelForObjectDetection.from_pretrained(
                str(self.best_ckpt_file),
            )
        elif self.task == TaskType.CLASSIFICATION:
            model = AutoModelForImageClassification.from_pretrained(
                str(self.best_ckpt_file),
            )

        model = ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
        )

        return model
