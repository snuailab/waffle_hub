from pathlib import Path
from typing import Union

import PIL.Image
from datasets.features.image import Image as ImageFeature
from waffle_utils.file import io

from datasets import (
    ClassLabel,
)
from datasets import Dataset as HFDataset
from datasets import (
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
from waffle_hub import TaskType
from waffle_hub.schema.fields import Annotation, Category, Image


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

        dataset[split] = HFDataset.from_generator(
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

        dataset[split] = HFDataset.from_generator(
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


def import_transformers(self, dataset_dir: str):
    """
    Import dataset from Transformers format

    Args:
        dataset_dir (str): Path to Transformers dataset directory
    """
    dataset = load_from_disk(dataset_dir)

    if isinstance(dataset, DatasetDict):
        is_splited = True
    elif isinstance(dataset, HFDataset):
        is_splited = False
    else:
        raise ValueError("dataset should be Dataset or DatasetDict")

    def _import(dataset: HFDataset, task: str, image_ids: list[int]):
        if task == TaskType.OBJECT_DETECTION:
            if not self.get_categories():
                categories = dataset.features["objects"].feature["category"].names
                for category_id, category_name in enumerate(categories):
                    category = Category.object_detection(
                        category_id=category_id + 1,
                        supercategory="object",
                        name=category_name,
                    )
                    self.add_categories([category])

            for data in dataset:
                data["image"].save(f"{self.raw_image_dir}/{data['image_id']}.jpg")
                image = Image.new(
                    image_id=data["image_id"],
                    file_name=f"{data['image_id']}.jpg",
                    width=data["width"],
                    height=data["height"],
                )
                self.add_images([image])

                annotation_ids = data["objects"]["id"]
                areas = data["objects"]["area"]
                category_ids = data["objects"]["category"]
                bboxes = data["objects"]["bbox"]

                for annotation_id, area, category_id, bbox in zip(
                    annotation_ids, areas, category_ids, bboxes
                ):
                    annotation = Annotation.object_detection(
                        annotation_id=annotation_id,
                        image_id=image.image_id,
                        category_id=category_id + 1,
                        area=area,
                        bbox=bbox,
                    )
                    self.add_annotations([annotation])

        elif task == TaskType.CLASSIFICATION:
            if not self.get_categories():
                categories = dataset.features["label"].names
                for category_id, category_name in enumerate(categories):
                    category = Category.classification(
                        category_id=category_id + 1,
                        supercategory="object",
                        name=category_name,
                    )
                    self.add_categories([category])

            for image_id, data in zip(image_ids, dataset):
                image_save_path = f"{self.raw_image_dir}/{image_id}.jpg"
                data["image"].save(image_save_path)
                pil_image = PIL.Image.open(image_save_path)
                width, height = pil_image.size
                image = Image.new(
                    image_id=image_id,
                    file_name=f"{image_id}.jpg",
                    width=width,
                    height=height,
                )
                self.add_images([image])

                annotation = Annotation.classification(
                    annotation_id=image_id,
                    image_id=image.image_id,
                    category_id=data["label"] + 1,
                )
                self.add_annotations([annotation])

        else:
            raise ValueError("task should be one of ['classification', 'object_detection']")

    if is_splited:
        start_num = 1
        for set_type, set in dataset.items():
            image_ids = list(range(start_num, set.num_rows + start_num))
            start_num += set.num_rows
            io.save_json(image_ids, self.set_dir / f"{set_type}.json", True)
            _import(set, self.task, image_ids)
    else:
        image_ids = list(range(1, dataset.num_rows + 1))
        _import(dataset, self.task, image_ids)
