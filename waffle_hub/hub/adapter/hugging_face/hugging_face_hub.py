"""
Hugging Face Hub
"""

from waffle_hub import get_installed_backend_version

BACKEND_NAME = "transformers"
BACKEND_VERSION = get_installed_backend_version(BACKEND_NAME)

import os
import warnings
from pathlib import Path
from typing import Union

import albumentations
import numpy as np
import torch
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)
from waffle_utils.file import io

from datasets import load_dataset
from waffle_hub.hub.base_hub import BaseHub, InferenceContext, TrainContext
from waffle_hub.hub.model.wrapper import ModelWrapper, ResultParser
from waffle_hub.utils.callback import TrainCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
        classes: Union[list[dict], list] = None,
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
            classes=classes,
            root_dir=root_dir,
        )

    @classmethod
    def new(
        cls,
        name: str,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        classes: Union[list[dict], list] = None,
        root_dir: str = None,
    ):

        return cls(
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            classes=classes,
            root_dir=root_dir,
        )

    def get_metrics(self) -> list[list[dict]]:
        # TODO: implement
        return []

    def on_train_start(self, ctx: TrainContext):
        # TODO: implement
        self.dataset = load_dataset("cppe-5")
        categories = (
            self.dataset["train"].features["objects"].feature["category"].names
        )
        id2label = {index: x for index, x in enumerate(categories, start=0)}
        label2id = {v: k for k, v in id2label.items()}

        remove_idx = [590, 821, 822, 875, 876, 878, 879]
        keep = [
            i for i in range(len(self.dataset["train"])) if i not in remove_idx
        ]
        self.dataset["train"] = self.dataset["train"].select(keep)

        ctx.pretrained_model = "facebook/detr-resnet-50"
        self.image_processor = AutoImageProcessor.from_pretrained(
            ctx.pretrained_model
        )

        # temp
        transform = albumentations.Compose(
            [
                albumentations.Resize(ctx.image_size, ctx.image_size),
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

        # transforming a batch

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

            return self.image_processor(
                images=images, annotations=targets, return_tensors="pt"
            )

        self.dataset["train"] = self.dataset["train"].with_transform(
            transform_aug_ann
        )
        print(self.dataset["train"][15])

        def collate_fn(batch):
            pixel_values = [item["pixel_values"] for item in batch]
            encoding = self.image_processor.pad_and_create_pixel_mask(
                pixel_values, return_tensors="pt"
            )
            labels = [item["labels"] for item in batch]
            batch = {}
            batch["pixel_values"] = encoding["pixel_values"]
            batch["pixel_mask"] = encoding["pixel_mask"]
            batch["labels"] = labels
            return batch

        self.collate_fn = collate_fn

        self.model = AutoModelForObjectDetection.from_pretrained(
            ctx.pretrained_model,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        self.training_args = TrainingArguments(
            output_dir=os.path.join(ctx.dataset_path, "training"),
            per_device_train_batch_size=ctx.batch_size,
            num_train_epochs=ctx.epochs,
            remove_unused_columns=False,
            push_to_hub=False,
        )
        print("on train start")

    def training(self, ctx: TrainContext, callback: TrainCallback):
        # TODO: implement
        print("train_loop")
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.collate_fn,
            train_dataset=self.dataset["train"],
            tokenizer=self.image_processor,
        )
        trainer.train()
        print("training")

    def on_train_end(self, ctx: TrainContext):
        # TODO: implement
        print("on train end")
