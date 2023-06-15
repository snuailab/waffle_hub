import warnings
from pathlib import Path
from typing import Union

from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.schema.fields.image import Image
from waffle_hub.utils.conversion import merge_multi_segment


def _check_valid_file_paths(images: list[Image]) -> bool:
    """Check file paths are valid
    If the file name includes the words "images" or "labels," an error occurs during training

    Args:
        images (list[Image]): List of Image

    Returns:
        bool: True if valid
    """
    for image in images:
        file_path = Path(image.file_name)
        if "images" in file_path.parts:
            raise ValueError(
                f"The file path '{file_path}' is not allowed. Please choose a file path that does not contain the word 'images'"
            )
        if "labels" in file_path.parts:
            raise ValueError(
                f"The file path '{file_path}' is not allowed. Please choose a file path that does not contain the word 'labels'"
            )
    else:
        return True


def _export_yolo_classification(
    self,
    export_dir: Path,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    unlabeled_ids: list,
):
    """Export dataset to YOLO format for classification task

    Args:
        export_dir (Path): Path to export directory
        train_ids (list): List of train ids
        val_ids (list): List of validation ids
        test_ids (list): List of test ids
        unlabeled_ids (list): List of unlabeled ids
    """
    io.make_directory(export_dir)

    for split, image_ids in zip(
        ["train", "val", "test", "unlabeled"],
        [train_ids, val_ids, test_ids, unlabeled_ids],
    ):
        if len(image_ids) == 0:
            continue

        split_dir = export_dir / split
        io.make_directory(split_dir)

        category_names = {c.category_id: c.name for c in self.get_categories()}
        for image in self.get_images(image_ids):
            image_path = self.raw_image_dir / image.file_name

            annotations = self.get_annotations(image.image_id)
            if len(annotations) > 1:
                warnings.warn(f"Multi label does not support yet. Skipping {image_path}.")
                continue
            category_id = annotations[0].category_id

            image_dst_path = split_dir / category_names[category_id] / image.file_name
            io.copy_file(image_path, image_dst_path, create_directory=True)


def _export_yolo_detection(
    self,
    export_dir: Path,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    unlabeled_ids: list,
):
    """Export dataset to YOLO format for detection task

    Args:
        export_dir (Path): Path to export directory
        train_ids (list): List of train ids
        val_ids (list): List of validation ids
        test_ids (list): List of test ids
        unlabeled_ids (list): List of unlabeled ids
    """
    io.make_directory(export_dir)

    for split, image_ids in zip(
        ["train", "val", "test", "unlabeled"],
        [train_ids, val_ids, test_ids, unlabeled_ids],
    ):
        if len(image_ids) == 0:
            continue

        image_dir = export_dir / split / "images"
        label_dir = export_dir / split / "labels"

        io.make_directory(image_dir)
        io.make_directory(label_dir)

        for image in self.get_images(image_ids):
            image_path = self.raw_image_dir / image.file_name
            image_dst_path = image_dir / image.file_name
            label_dst_path = (label_dir / image.file_name).with_suffix(".txt")
            io.copy_file(image_path, image_dst_path, create_directory=True)

            W = image.width
            H = image.height

            annotations = self.get_annotations(image.image_id)
            label_txts = []
            for annotation in annotations:
                x1, y1, w, h = annotation.bbox
                x1, w = x1 / W, w / W
                y1, h = y1 / H, h / H
                cx, cy = x1 + w / 2, y1 + h / 2

                category_id = annotation.category_id - 1

                label_txts.append(f"{category_id} {cx} {cy} {w} {h}")

            io.make_directory(label_dst_path.parent)
            with open(label_dst_path, "w") as f:
                f.write("\n".join(label_txts))


def _export_yolo_segmentation(
    self,
    export_dir: Path,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    unlabeled_ids: list,
):
    io.make_directory(export_dir)

    for split, image_ids in zip(
        ["train", "val", "test", "unlabeled"],
        [train_ids, val_ids, test_ids, unlabeled_ids],
    ):
        if len(image_ids) == 0:
            continue

        image_dir = export_dir / split / "images"
        label_dir = export_dir / split / "labels"

        io.make_directory(image_dir)
        io.make_directory(label_dir)

        for image in self.get_images(image_ids):
            image_path = self.raw_image_dir / image.file_name
            image_dst_path = image_dir / image.file_name
            label_dst_path = (label_dir / image.file_name).with_suffix(".txt")
            io.copy_file(image_path, image_dst_path, create_directory=True)

            W = image.width
            H = image.height

            annotations = self.get_annotations(image.image_id)
            label_txts = []
            for annotation in annotations:
                x1, y1, w, h = annotation.bbox
                x1, w = x1 / W, w / W
                y1, h = y1 / H, h / H

                category_id = annotation.category_id - 1

                segment = merge_multi_segment(annotation.segmentation, (W, H))
                segment[0::2] = [x / W for x in segment[0::2]]
                segment[1::2] = [y / H for y in segment[1::2]]
                segment = " ".join(map(str, segment))

                label_txts.append(f"{category_id} {segment}")

            io.make_directory(label_dst_path.parent)
            with open(label_dst_path, "w") as f:
                f.write("\n".join(label_txts))


def export_yolo(self, export_dir: Union[str, Path]) -> str:
    """Export dataset to YOLO format

    Args:
        export_dir (Union[str, Path]): Path to export directory

    Returns:
        str: Path to export directory
    """
    _check_valid_file_paths(self.get_images())

    export_dir = Path(export_dir)

    train_ids, val_ids, test_ids, _ = self.get_split_ids()

    if self.task == TaskType.CLASSIFICATION:
        _export_yolo_classification(self, export_dir, train_ids, val_ids, test_ids, [])
    elif self.task == TaskType.OBJECT_DETECTION:
        _export_yolo_detection(self, export_dir, train_ids, val_ids, test_ids, [])
    elif self.task == TaskType.INSTANCE_SEGMENTATION:
        _export_yolo_segmentation(self, export_dir, train_ids, val_ids, test_ids, [])
    else:
        raise ValueError(f"Unsupported task type: {self.task}")

    io.save_yaml(
        {
            "path": str(export_dir.absolute()),
            "train": "train",
            "val": "val",
            "test": "test",
            "names": {category.category_id - 1: category.name for category in self.get_categories()},
        },
        export_dir / "data.yaml",
    )

    return str(export_dir)
