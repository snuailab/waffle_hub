from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Union

import albumentations
import evaluate
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
    compute_metrics: Callable = None


class TrainInputHelper(ABC):
    """
    This class is designed to assist with passing arguments to the Hugging Face's Trainer function.
    """

    def __init__(
        self, pretrained_model: str, image_size: Union[int, list[int]]
    ) -> None:
        self.pretrained_model = pretrained_model
        self.image_size = image_size

        self.image_processor: AutoImageProcessor = self.get_image_processor(
            self.pretrained_model
        )
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

    @abstractmethod
    def get_model(self, categories: list) -> _BaseAutoModelClass:
        """
        Returns an instance of a pre-trained model.

        Args:
            categories (int): The number of output categories the model will predict.

        Returns:
            PreTrainedModel: The pre-trained model.

        """
        pass

    @abstractmethod
    def get_categories(self, features: Features) -> list:
        """
        Returns the number of categories for a given dataset.

        Args:
            features (Features): The dataset features.

        Returns:
            int: The number of categories.

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
    def __init__(
        self, pretrained_model: str, image_size: Union[int, list[int]]
    ) -> None:
        super().__init__(pretrained_model, image_size)

    def get_transforms(self) -> Callable:
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

        def transforms(examples: dict) -> dict:
            examples["pixel_values"] = [
                _transforms(img.convert("RGB")) for img in examples["image"]
            ]
            del examples["image"]
            if "image_id" in examples.keys():
                del examples["image_id"]
                del examples["width"]
                del examples["height"]
            return examples

        return transforms

    def get_collator(self) -> Callable:
        return DefaultDataCollator(return_tensors="pt")

    def get_model(self, categories: list) -> _BaseAutoModelClass:
        id2label = {index: x for index, x in enumerate(categories, start=0)}
        return AutoModelForImageClassification.from_pretrained(
            self.pretrained_model,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        )

    def get_categories(self, features: Features) -> list:
        return features["label"].names

    def get_compute_metrics(self) -> Callable:
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred: np.ndarray) -> dict:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            return metric.compute(predictions=predictions, references=labels)

        return compute_metrics


class ObjectDetectionInputHelper(TrainInputHelper):
    def __init__(
        self, pretrained_model: str, image_size: Union[int, list[int]]
    ) -> None:
        super().__init__(pretrained_model, image_size)

    def get_transforms(self) -> Callable:

        size = (
            self.image_processor.size["shortest_edge"]
            if "shortest_edge" in self.image_processor.size
            else (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
            )
        )

        _transforms = albumentations.Compose(
            [
                albumentations.Resize(size, size),
                albumentations.HorizontalFlip(p=1.0),
                albumentations.RandomBrightnessContrast(p=1.0),
            ],
            bbox_params=albumentations.BboxParams(
                format="coco", label_fields=["category"]
            ),
        )

        def formatted_anns(
            image_id: int, category: list, area: list, bbox: list
        ) -> list[dict]:
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
                for id_, cat_, ar_, box_ in zip(
                    image_ids, categories, area, bboxes
                )
            ]

            return self.image_processor(
                images=images, annotations=targets, return_tensors="pt"
            )

        return transforms

    def get_collator(self) -> Callable:
        def collate_fn(batch: list[dict]) -> dict:
            pixel_values = [item["pixel_values"] for item in batch]
            encoding = self.image_processor.pad_and_create_pixel_mask(
                pixel_values, return_tensors="pt"
            )
            labels = [item["labels"] for item in batch]

            new_batch = {}
            new_batch["pixel_values"] = encoding["pixel_values"]
            new_batch["pixel_mask"] = encoding["pixel_mask"]
            new_batch["labels"] = labels
            return new_batch

        return collate_fn

    def get_model(self, categories) -> _BaseAutoModelClass:
        id2label = {index: x for index, x in enumerate(categories, start=0)}
        label2id = {x: index for index, x in enumerate(categories, start=0)}
        model = AutoModelForObjectDetection.from_pretrained(
            self.pretrained_model,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        return model

    def get_categories(self, features: Features) -> list:
        return features["objects"].feature["category"].names
