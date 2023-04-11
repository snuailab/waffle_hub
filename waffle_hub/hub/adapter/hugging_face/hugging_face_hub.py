"""
Hugging Face Hub
"""

from waffle_hub import get_installed_backend_version

BACKEND_NAME = "transformers"
BACKEND_VERSION = get_installed_backend_version(BACKEND_NAME)

import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Union

import albumentations
import numpy as np
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
    pipeline,
)

from datasets import DatasetDict, load_from_disk
from waffle_hub.hub.base_hub import BaseHub
from waffle_hub.schema.configs import TrainConfig
from waffle_hub.utils.callback import TrainCallback


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
            "DETA": ["base", "large"],
            "DETR": ["base", "large"],
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

        self.train_input = TrainInput()

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

    def get_metrics(self) -> list[list[dict]]:
        # TODO: implement
        return []

    def on_train_start(self, cfg: TrainConfig):
        # TODO: model choice

        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device

        cfg.pretrained_model = f"facebook/{self.model_type.lower()}-resnet-50"

        dataset = load_from_disk(cfg.dataset_path)

        categories = (
            dataset["train"].features["objects"].feature["category"].names
        )
        id2label = {index: x for index, x in enumerate(categories, start=0)}
        label2id = {v: k for k, v in id2label.items()}

        image_processor = AutoImageProcessor.from_pretrained(
            cfg.pretrained_model
        )

        transform = albumentations.Compose(
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

        def transform_aug_ann(examples):
            image_ids = examples["image_id"]
            images, bboxes, area, categories = [], [], [], []
            for image, objects in zip(examples["image"], examples["objects"]):
                image = np.array(image.convert("RGB"))[:, :, ::-1]
                out = transform(
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

        dataset["train"] = dataset["train"].with_transform(transform_aug_ann)

        self.train_input.train_data = dataset

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

        if self.task == "object_detection":
            self.train_input.model = (
                AutoModelForObjectDetection.from_pretrained(
                    cfg.pretrained_model,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,
                )
            )

        self.train_input.training_args = TrainingArguments(
            output_dir=str(self.artifact_dir),
            per_device_train_batch_size=cfg.batch_size,
            num_train_epochs=cfg.epochs,
            remove_unused_columns=False,
            push_to_hub=False,
        )

    def training(self, cfg: TrainConfig, callback: TrainCallback):
        trainer = Trainer(
            model=self.train_input.model,
            args=self.train_input.training_args,
            data_collator=self.train_input.collate_fn,
            train_dataset=self.train_input.train_data["train"],
            tokenizer=self.train_input.image_processor,
        )
        trainer.train()
        trainer.save_model(str(self.artifact_dir / "weights"))

    def on_train_end(self, cfg: TrainConfig):
        return super().on_train_end(cfg)

    # Inference Hook
    def get_model(self):
        model = pipeline(model="")  # ????
        return model
