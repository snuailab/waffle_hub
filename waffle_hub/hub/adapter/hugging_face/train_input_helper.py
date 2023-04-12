from dataclasses import dataclass, field
from typing import Callable, Union

import albumentations
import numpy as np
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    DefaultDataCollator,
    TrainingArguments,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from datasets import (
    DatasetDict,
    Features,
)


@dataclass
class TrainInput:
    dataset: DatasetDict = field(default_factory=DatasetDict)
    image_processor: AutoImageProcessor = None
    training_args: TrainingArguments = None
    collator: Callable = None
    model: _BaseAutoModelClass = None


class TrainInputHelper:
    def __init__(self, pretrained_model: str) -> None:
        self.pretrained_model = pretrained_model

        self.image_processor = self.get_image_processor(self.pretrained_model)
        self.train_input = TrainInput()

    def get_image_processor(self, pretrained_model: str):
        return AutoImageProcessor.from_pretrained(pretrained_model)

    def get_transforms(self):
        pass

    def get_collator(self):
        pass

    def get_model(self, categories):
        pass

    def get_categories(self, features: Features):
        pass

    def get_train_input(self):
        train_input = TrainInput()
        train_input.image_processor = self.image_processor
        train_input.collator = self.get_collator()
        return train_input


class ClassifierInputHelper(TrainInputHelper):
    def __init__(self, pretrained_model: str) -> None:
        super().__init__(pretrained_model)

    def get_transforms(self):
        normalize = T.Normalize(
            mean=self.image_processor.image_mean,
            std=self.image_processor.image_std,
        )
        size = (
            self.image_processor.size["shortest_edge"]
            if "shortest_edge" in self.image_processor.size
            else (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
            )
        )

        _transforms = T.Compose(
            [T.RandomResizedCrop(size), T.ToTensor(), normalize]
        )

        def transforms(examples):
            examples["pixel_values"] = [
                _transforms(img.convert("RGB")) for img in examples["image"]
            ]
            del examples["image"]
            return examples

        return transforms

    def get_collator(self):
        return DefaultDataCollator(return_tensors="pt")

    def get_model(self, categories: list) -> _BaseAutoModelClass:
        id2label = {index: x for index, x in enumerate(categories, start=0)}
        return AutoModelForImageClassification.from_pretrained(
            self.pretrained_model,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        )

    def get_categories(self, features: Features):
        return features["label"].names


class ObjectDetectionInputHelper(TrainInputHelper):
    def __init__(
        self, pretrained_model: str, image_size: Union[int, list[int]]
    ) -> None:
        super().__init__(pretrained_model)
        self.image_size = image_size

    def get_transforms(self):
        _transforms = albumentations.Compose(
            [
                albumentations.Resize(self.image_size, self.image_size),
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
            for image, objects in zip(examples["image"], examples["objects"]):
                image = np.array(image.convert("RGB"))[:, :, ::-1]
                out = _transforms(
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

        return transforms

    def get_collator(self):
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

        return collate_fn

    def get_model(self, categories):
        id2label = {index: x for index, x in enumerate(categories, start=0)}
        label2id = {x: index for index, x in enumerate(categories, start=0)}
        model = AutoModelForObjectDetection.from_pretrained(
            self.pretrained_model,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        return model

    def get_categories(self, features: Features):
        return features["objects"].feature["category"].names
