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
from waffle_hub.hub.adapter.hugging_face.train_input_helper import (
    ClassifierInputHelper,
    ObjectDetectionInputHelper,
)
from waffle_hub.hub.base_hub import BaseHub
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import TrainConfig
from waffle_hub.utils.callback import TrainCallback


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

    # Common
    MODEL_TYPES = {
        "object_detection": {
            "DETA": {
                "base": "jozhang97/deta-resnet-50",
            },
            "DETR": {
                "base": "facebook/detr-resnet-50",
                "large": "facebook/detr-resnet-101",
            },
            "YOLOS": {
                "tiny": "hustvl/yolos-tiny",
            },
        },
        "classification": {
            "ViT": {
                "tiny": "WinKawaks/vit-tiny-patch16-224",
                "base": "google/vit-base-patch16-224",
            }
        },
    }

    DEFAULT_PARAMAS = {
        "object_detection": {
            "epochs": 50,
            "image_size": [800, 800],
            "learning_rate": 5e-05,
            "letter_box": True,  # TODO: implement letter_box
            "batch_size": 16,
        },
        "classification": {
            "epochs": 50,
            "image_size": [224, 224],
            "learning_rate": 5e-05,
            "letter_box": False,
            "batch_size": 16,
        },
    }

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

        self.pretrained_model: str = self.MODEL_TYPES[self.task][self.model_type][self.model_size]
        self.best_ckpt_dir = self.hub_dir / "weights" / "best_ckpt"
        self.train_input = None

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
        cfg.pretrained_model = self.pretrained_model
        for k, v in cfg.to_dict().items():
            if v is None:
                setattr(cfg, k, self.DEFAULT_PARAMAS[self.task][k])

        # setting
        if cfg.device != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        dataset = load_from_disk(cfg.dataset_path)

        if self.task == "classification":
            helper = ClassifierInputHelper(cfg.pretrained_model, cfg.image_size)
            self.train_input = helper.get_train_input()
            categories = dataset["train"].features["label"].names
            id2label = {index: x for index, x in enumerate(categories, start=0)}
            self.train_input.model = AutoModelForImageClassification.from_pretrained(
                self.pretrained_model,
                num_labels=len(id2label),
                ignore_mismatched_sizes=True,
            )

        elif self.task == "object_detection":
            helper = ObjectDetectionInputHelper(cfg.pretrained_model, cfg.image_size)
            self.train_input = helper.get_train_input()
            categories = dataset["train"].features["objects"].feature["category"].names
            id2label = {index: x for index, x in enumerate(categories, start=0)}
            label2id = {x: index for index, x in enumerate(categories, start=0)}
            self.train_input.model = AutoModelForObjectDetection.from_pretrained(
                self.pretrained_model,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            raise NotImplementedError

        transforms = helper.get_transforms()
        dataset["train"] = dataset["train"].with_transform(transforms)
        dataset["val"] = dataset["val"].with_transform(transforms)
        self.train_input.dataset = dataset

        self.train_input.training_args = TrainingArguments(
            output_dir=str(self.artifact_dir),
            per_device_train_batch_size=cfg.batch_size,
            num_train_epochs=cfg.epochs,
            remove_unused_columns=False,
            push_to_hub=False,
            logging_strategy="epoch" if cfg.verbose else "no",
            evaluation_strategy="epoch",
            learning_rate=cfg.learning_rate,
            dataloader_num_workers=cfg.workers,
            seed=cfg.seed,
            greater_is_better=True,
        )

    def training(self, cfg: TrainConfig, callback: TrainCallback):

        trainer = Trainer(
            model=self.train_input.model,
            args=self.train_input.training_args,
            data_collator=self.train_input.collator,
            train_dataset=self.train_input.dataset["train"],
            eval_dataset=self.train_input.dataset["val"],
            tokenizer=self.train_input.image_processor,
            compute_metrics=self.train_input.compute_metrics,
        )
        trainer.add_callback(CustomCallback(trainer))
        trainer.train()
        trainer.save_model(str(self.artifact_dir / "weights"))
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
            self.artifact_dir / "weights",
            self.best_ckpt_dir,
            create_directory=True,
        )
        io.save_json(self.get_metrics(), self.metric_file)

    def get_preprocess(self, task, pretrained_model: str) -> Callable:
        image_processer = AutoImageProcessor.from_pretrained(pretrained_model)

        normalize = T.Normalize(image_processer.image_mean, image_processer.image_std, inplace=True)

        def preprocess(x, *args, **kwargs):
            return normalize(x)

        return preprocess

    def get_postprocess(self, task: str) -> Callable:
        if task == "classification":

            def inner(x: ModelOutput, *args, **kwargs) -> torch.Tensor:
                return [x.logits]

        elif task == "object_detection":

            def inner(x: ModelOutput, *args, **kwargs) -> torch.Tensor:

                if x.logits.shape[-1] != len(self.categories):
                    x.logits = x.logits[:, :, :-1]  # remove background
                confidences, category_ids = torch.max(x.logits[:, :, :], dim=-1)
                cxcywh = x.pred_boxes[:, :, :4]
                cx, cy, w, h = torch.unbind(cxcywh, dim=-1)
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                return xyxy, confidences, category_ids

        else:
            raise NotImplementedError(f"Task {task} is not implemented.")

        return inner

    def get_model(self) -> ModelWrapper:
        self.check_train_sanity()

        # get adapt functions
        preprocess = self.get_preprocess(self.task, self.pretrained_model)
        postprocess = self.get_postprocess(self.task)

        # get model
        if self.task == "object_detection":
            model = AutoModelForObjectDetection.from_pretrained(
                str(self.best_ckpt_dir),
            )
        elif self.task == "classification":
            model = AutoModelForImageClassification.from_pretrained(
                str(self.best_ckpt_dir),
            )

        model = ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
        )

        return model

    def check_train_sanity(self) -> bool:
        if not (self.model_config_file.exists() and self.best_ckpt_dir.exists()):
            raise FileNotFoundError("Train first! hub.train(...).")
        return True
