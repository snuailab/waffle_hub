"""
Hugging Face Hub
"""

from waffle_hub import get_installed_backend_version

BACKEND_NAME = "transformers"
BACKEND_VERSION = get_installed_backend_version(BACKEND_NAME)

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

import albumentations
import cv2
import numpy as np
import torch
import tqdm
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    DefaultDataCollator,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    pipeline,
)
from waffle_utils.file import io

from datasets import DatasetDict, load_from_disk
from waffle_hub.hub.base_hub import BaseHub
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import InferenceConfig, TrainConfig
from waffle_hub.utils.callback import InferenceCallback, TrainCallback
from waffle_hub.utils.data import ImageDataset
from waffle_hub.utils.draw import draw_results


@dataclass
class TrainInput:
    train_data: DatasetDict = field(default_factory=DatasetDict)
    image_processor: AutoImageProcessor = None
    training_args: TrainingArguments = None
    collate_fn: Callable = None
    model: AutoModelForObjectDetection = None


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
        },
        "classification": {
            "ViT": {
                "base": "google/vit-base-patch16-224",
            }
        },
    }

    DEFAULT_PARAMAS = {
        "object_detection": {
            "epochs": 50,
            "image_size": [640, 640],
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
            raise ValueError(
                f"you've loaded {backend}. backend must be {BACKEND_NAME}"
            )

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

        self.pretrained_model: str = self.MODEL_TYPES[self.task][
            self.model_type
        ][self.model_size]
        self.train_input = TrainInput()
        self.best_ckpt_dir = self.hub_dir / "weights" / "best_ckpt"

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

        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device

        dataset = load_from_disk(cfg.dataset_path)

        if self.task == "object_detection":
            categories = (
                dataset["train"].features["objects"].feature["category"].names
            )
        elif self.task == "classification":
            categories = dataset["train"].features["label"].names

        id2label = {index: x for index, x in enumerate(categories, start=0)}
        label2id = {v: k for k, v in id2label.items()}

        image_processor = AutoImageProcessor.from_pretrained(
            cfg.pretrained_model
        )

        if self.task == "object_detection":
            _transform = albumentations.Compose(
                [
                    albumentations.Resize(cfg.image_size, cfg.image_size),
                    albumentations.HorizontalFlip(p=1.0),
                    albumentations.RandomBrightnessContrast(p=1.0),
                ],
                bbox_params=albumentations.BboxParams(
                    format="coco", label_fields=["category"]
                ),
            )

            def formatted_anns(image_id, category, area, bbox):
                annotations = []
                for i in range(0, len(category)):
                    new_ann = {
                        "image_id": image_id,
                        "category_id": category[i],
                        "isCrowd": 0,
                        "area": area[i],
                        "bbox": list(bbox[i]),
                    }
                    annotations.append(new_ann)

                return annotations

            def transforms(examples):
                image_ids = examples["image_id"]
                images, bboxes, area, categories = [], [], [], []
                for image, objects in zip(
                    examples["image"], examples["objects"]
                ):
                    image = np.array(image.convert("RGB"))[:, :, ::-1]
                    out = _transform(
                        image=image,
                        bboxes=objects["bbox"],
                        category=objects["category"],
                    )

                    area.append(objects["area"])
                    images.append(out["image"])
                    bboxes.append(out["bboxes"])
                    categories.append(out["category"])

                targets = [
                    {
                        "image_id": id_,
                        "annotations": formatted_anns(id_, cat_, ar_, box_),
                    }
                    for id_, cat_, ar_, box_ in zip(
                        image_ids, categories, area, bboxes
                    )
                ]

                return image_processor(
                    images=images, annotations=targets, return_tensors="pt"
                )

            def collate_fn(batch):
                pixel_values = [item["pixel_values"] for item in batch]
                encoding = image_processor.pad_and_create_pixel_mask(
                    pixel_values, return_tensors="pt"
                )
                labels = [item["labels"] for item in batch]
                batch = {}
                batch["pixel_values"] = encoding["pixel_values"]
                batch["pixel_mask"] = encoding["pixel_mask"]
                batch["labels"] = labels
                return batch

            self.train_input.collate_fn = collate_fn

            self.train_input.model = (
                AutoModelForObjectDetection.from_pretrained(
                    cfg.pretrained_model,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,
                )
            )

        elif self.task == "classification":
            normalize = T.Normalize(
                mean=image_processor.image_mean, std=image_processor.image_std
            )
            size = (
                image_processor.size["shortest_edge"]
                if "shortest_edge" in image_processor.size
                else (
                    image_processor.size["height"],
                    image_processor.size["width"],
                )
            )
            _transforms = T.Compose(
                [T.RandomResizedCrop(size), T.ToTensor(), normalize]
            )

            def transforms(examples):
                examples["pixel_values"] = [
                    _transforms(img.convert("RGB"))
                    for img in examples["image"]
                ]
                del examples["image"]
                return examples

            self.train_input.collate_fn = DefaultDataCollator(
                return_tensors="pt"
            )

            self.train_input.model = (
                AutoModelForImageClassification.from_pretrained(
                    cfg.pretrained_model,
                    num_labels=len(id2label),
                    ignore_mismatched_sizes=True,
                )
            )

        dataset["train"] = dataset["train"].with_transform(transforms)
        dataset["val"] = dataset["val"].with_transform(transforms)

        self.train_input.train_data = dataset

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
            data_collator=self.train_input.collate_fn,
            train_dataset=self.train_input.train_data["train"],
            eval_dataset=self.train_input.train_data["val"],
            tokenizer=self.train_input.image_processor,
        )
        trainer.train()
        trainer.save_model(str(self.artifact_dir / "weights"))
        self.train_log = trainer.state.log_history

    def get_metrics(self) -> list[list[dict]]:
        metrics = []

        for epoch in range(
            0, len(self.train_log) - 1, 2
        ):  # last is runtime info
            train_log, eval_log = (
                self.train_log[epoch],
                self.train_log[epoch + 1],
            )
            epoch_metrics = []

            train_log.update(eval_log)
            for key, value in train_log.items():
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

    def get_preprocess(self, task, pretrained_model: str):
        image_processer = AutoImageProcessor.from_pretrained(pretrained_model)

        normalize = T.Normalize(
            image_processer.image_mean, image_processer.image_std, inplace=True
        )

        def preprocess(x, *args, **kwargs):
            return normalize(x)

        return preprocess

    def get_postprocess(self, task):
        if task == "classification":

            def inner(x: torch.Tensor, *args, **kwargs):
                return [x.logits]

        elif task == "object_detection":

            def inner(x: torch.Tensor, *args, **kwargs):

                confidences, category_ids = torch.max(
                    x.logits[:, :, :-1], dim=-1
                )
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

    def get_model(self):
        """Get model.
        Returns:
            ModelWrapper: Model wrapper
        """
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
        """Check if all essential files are exist.

        Returns:
            bool: True if all files are exist else False
        """
        if not (
            self.model_config_file.exists()
            and self.best_ckpt_dir.exists()
            # and self.last_ckpt_file.exists()
        ):
            raise FileNotFoundError("Train first! hub.train(...).")
        return True
