from pathlib import Path
from typing import Union

import PIL.Image
from datasets.features.image import Image as ImageFeature

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from waffle_hub import TaskType
from waffle_hub.schema.fields import Image


def _export_transformers_classification(
    self,
    export_dir: Path,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    unlabeled_ids: list,
):
    """Export dataset to Transformers format for classification task

    Args:
        export_dir (Path): Path to export directory
        train_ids (list): List of train ids
        val_ids (list): List of validation ids
        test_ids (list): List of test ids
        unlabeled_ids (list): List of unlabeled ids
    """
    category_names = self.get_category_names()
    features = Features(
        {
            "image": ImageFeature(),
            "label": ClassLabel(names=category_names),
        }
    )

    def _export(images: list[Image]):
        for image in images:
            annotation = self.get_annotations(image.image_id)[0]
            image_path = self.raw_image_dir / image.file_name
            yield {
                "image": PIL.Image.open(image_path).convert("RGB"),
                "label": category_names[annotation.category_id - 1],
            }

    dataset = {}
    for split, image_ids in zip(
        ["train", "val", "test", "unlabeled"],
        [train_ids, val_ids, test_ids, unlabeled_ids],
    ):
        if len(image_ids) == 0:
            continue

        dataset[split] = Dataset.from_generator(
            lambda: _export(self.get_images(image_ids)), features=features
        )

    dataset = DatasetDict(dataset)
    dataset.save_to_disk(export_dir)


def _export_transformers_detection(
    self,
    export_dir: Path,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    unlabeled_ids: list,
):
    """Export dataset to Transformers format for detection task

    Args:
        export_dir (Path): Path to export directory
        train_ids (list): List of train ids
        val_ids (list): List of validation ids
        test_ids (list): List of test ids
        unlabeled_ids (list): List of unlabeled ids
    """
    category_names = self.get_category_names()
    features = Features(
        {
            "image": ImageFeature(),
            "image_id": Value("int32"),
            "width": Value("int32"),
            "height": Value("int32"),
            "objects": Sequence(
                {
                    "id": Value("int32"),
                    "area": Value("int32"),
                    "category": ClassLabel(names=category_names),
                    "bbox": Sequence(Value("float32")),
                }
            ),
        }
    )

    def _export(images: list[Image]):
        for image in images:
            # objects
            annotations = self.get_annotations(image.image_id)
            objects = []
            for annotation in annotations:
                objects.append(
                    {
                        "id": annotation.annotation_id,
                        "area": annotation.area,
                        "category": category_names[annotation.category_id - 1],
                        "bbox": annotation.bbox,
                    }
                )

            # image
            image_path = self.raw_image_dir / image.file_name
            pil_image = PIL.Image.open(image_path).convert("RGB")
            yield {
                "image": pil_image,
                "image_id": image.image_id,
                "width": image.width,
                "height": image.height,
                "objects": objects,
            }

    dataset = {}
    for split, image_ids in zip(
        ["train", "val", "test", "unlabeled"],
        [train_ids, val_ids, test_ids, unlabeled_ids],
    ):
        if len(image_ids) == 0:
            continue

        dataset[split] = Dataset.from_generator(
            lambda: _export(self.get_images(image_ids)), features=features
        )

    dataset = DatasetDict(dataset)
    dataset.save_to_disk(export_dir)


def export_transformers(self, export_dir: Union[str, Path]) -> str:
    """Export dataset to Transformers format

    Args:
        export_dir (str): Path to export directory

    Returns:
        str: Path to export directory
    """
    export_dir = Path(export_dir)

    train_ids, val_ids, test_ids, _ = self.get_split_ids()

    if self.task == TaskType.CLASSIFICATION:
        _export_transformers_classification(self, export_dir, train_ids, val_ids, test_ids, [])
    elif self.task == TaskType.OBJECT_DETECTION:
        _export_transformers_detection(self, export_dir, train_ids, val_ids, test_ids, [])
    else:
        raise ValueError(f"Unsupported task type: {self.task}")

    return str(export_dir)
