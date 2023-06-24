import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Union

import albumentations
import evaluate
import numpy as np
import torch
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
    compute_metrics: Callable = None


class customTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        self._device = kwargs.pop("device")
        super().__init__(*args, **kwargs)

    @property
    def device(self) -> "torch.device":
        if self._device == "cpu":
            return torch.device("cpu")
        elif "," in self._device:
            return torch.device("cuda")
        else:
            return torch.device(f"cuda:{self._device}")

    @property
    def n_gpu(self) -> int:
        if self._device == "cpu":
            return 0
        elif "," in self._device:
            return len(self._device.split(","))
        else:
            return 1


class TrainInputHelper(ABC):
    """
    This class is designed to assist with passing arguments to the Transformers's Trainer function.
    """

    def __init__(self, pretrained_model: str, image_size: Union[int, list[int]]) -> None:
        self.pretrained_model = pretrained_model
        self.image_size = image_size

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)

        self.image_processor: AutoImageProcessor = self.get_image_processor(self.pretrained_model)
        self.train_input = TrainInput()

    def get_image_processor(self, pretrained_model: str) -> AutoImageProcessor:
        """
        Returns an instance of AutoImageProcessor for a given pre-trained model.

        Args:
            pretrained_model (str): The name or path of the pre-trained image processing model.

        Returns:
            AutoImageProcessor: The image processor model.

        """

        return AutoImageProcessor.from_pretrained(pretrained_model)

    @abstractmethod
    def get_transforms(self) -> Callable:
        """
        Returns a list of data augmentations to apply to the input data during training.

        Returns:
            List: A list of data augmentations to apply.

        """
        pass

    @abstractmethod
    def get_collator(self) -> Callable:
        """
        Returns an instance of DataCollator responsible for collating input data into batches.

        Returns:
            DataCollator: The data collator.

        """
        pass

    def get_compute_metrics(self) -> Callable:
        """
        Returns a function that computes metrics.

        Returns:
            Callable: The function that computes metrics.

        """
        return None

    def get_train_input(self) -> TrainInput:
        """
        Returns a TrainInput object containing all necessary input data for training.

        Returns:
            TrainInput: The training input data.

        """
        train_input = TrainInput()
        train_input.image_processor = self.image_processor
        train_input.collator = self.get_collator()
        train_input.compute_metrics = self.get_compute_metrics()
        return train_input


class ClassifierInputHelper(TrainInputHelper):
    def __init__(self, pretrained_model: str, image_size: Union[int, list[int]]) -> None:
        super().__init__(pretrained_model, image_size)

    def get_transforms(self) -> Callable:
        normalize = T.Normalize(
            mean=self.image_processor.image_mean,
            std=self.image_processor.image_std,
        )

        size = (
            (self.image_processor.size["shortest_edge"],) * 2
            if "shortest_edge" in self.image_processor.size
            else (
                self.image_processor.size["width"],
                self.image_processor.size["height"],
            )
        )
        if tuple(size) != tuple(self.image_size):
            raise ValueError(
                f"pretrained model's image size is {size}, but you set {self.image_size}."
            )

        _transforms = T.Compose(
            [T.RandomResizedCrop(self.image_size[::-1]), T.ToTensor(), normalize]
        )

        def transforms(examples: dict) -> dict:
            examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
            del examples["image"]
            return examples

        return transforms

    def get_collator(self) -> Callable:
        return DefaultDataCollator(return_tensors="pt")

    def get_compute_metrics(self) -> Callable:
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred: np.ndarray) -> dict:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            return metric.compute(predictions=predictions, references=labels)

        return compute_metrics


class ObjectDetectionInputHelper(TrainInputHelper):
    def __init__(self, pretrained_model: str, image_size: Union[int, list[int]]) -> None:
        super().__init__(pretrained_model, image_size)

    def get_transforms(self) -> Callable:
        if "shortest_edge" in self.image_processor.size:
            shortest_size = self.image_processor.size["shortest_edge"]
            longest_size = self.image_processor.size["longest_edge"]
            if min(self.image_size) < shortest_size:
                warnings.warn(
                    f"pretrained model's shortest edge is {shortest_size}, but you set {self.image_size}."
                )
                self.image_processor.size["shortest_edge"] = min(self.image_size)
            if max(self.image_size) > longest_size:
                warnings.warn(
                    f"pretrained model's longest edge is {longest_size}, but you set {self.image_size}."
                )
                self.image_processor.size["longest_edge"] = max(self.image_size)
        else:
            size = (self.image_processor.size["width"], self.image_processor.size["height"])

            if tuple(size) != tuple(self.image_size):
                warnings.warn(
                    f"pretrained model's image size is {size}, but you set {self.image_size}."
                )
                self.image_processor.size["width"] = self.image_size[0]
                self.image_processor.size["height"] = self.image_size[1]

        _transforms = albumentations.Compose(
            [
                albumentations.Resize(width=self.image_size[0], height=self.image_size[1]),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightnessContrast(p=0.5),
            ],
            bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
        )

        def formatted_anns(image_id: int, category: list, area: list, bbox: list) -> list[dict]:
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

        def transforms(examples: dict) -> dict:
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
                for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
            ]

            return self.image_processor(images=images, annotations=targets, return_tensors="pt")

        return transforms

    def get_collator(self) -> Callable:
        def collate_fn(batch: list[dict]) -> dict:
            pixel_values = [item["pixel_values"] for item in batch]
            labels = [item["labels"] for item in batch]

            return {
                "pixel_values": torch.stack(pixel_values),
                "labels": labels,
            }

        return collate_fn
