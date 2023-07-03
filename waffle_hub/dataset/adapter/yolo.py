import warnings
from pathlib import Path
from typing import Union

import cv2
from waffle_utils.file import io, search

from waffle_hub import TaskType
from waffle_hub.schema.fields import Annotation, Category, Image
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


def _import_yolo_classification(self, set_dir: Path, image_ids: list[int], info: dict):
    # categories
    if not self.get_category_names():
        for category_id, category_name in info["names"].items():
            self.add_categories(
                [
                    Category.classification(
                        category_id=category_id + 1,
                        name=category_name,
                    )
                ]
            )
    name2id = {v: k for k, v in info["names"].items()}

    for image_id, image_path in zip(image_ids, search.get_image_files(set_dir)):
        category_name_index = image_path.parts.index(set_dir.stem) + 1
        category_name = image_path.parts[category_name_index]
        category_id = name2id[category_name] + 1

        # image
        img = cv2.imread(str(image_path))
        height, width, _ = img.shape
        image = Image.new(
            image_id=image_id,
            file_name=f"{image_id}{image_path.suffix}",
            width=width,
            height=height,
        )
        self.add_images([image])

        # annotation
        annotation = Annotation.classification(
            annotation_id=image_id,
            image_id=image_id,
            category_id=category_id,
        )
        self.add_annotations([annotation])

        # raw
        dst = self.raw_image_dir / f"{image_id}{image_path.suffix}"
        io.copy_file(image_path, dst)


def _import_yolo_object_detection(self, set_dir: Path, image_ids: list[int], info: dict):
    image_dir = set_dir / "images"
    label_dir = set_dir / "labels"

    if not image_dir.exists():
        warnings.warn(f"{image_dir} does not exist.")
        return
    if not label_dir.exists():
        warnings.warn(f"{label_dir} does not exist.")
        return

    # categories
    if not self.get_category_names():
        for category_id, category_name in info["names"].items():
            self.add_categories(
                [
                    Category.object_detection(
                        category_id=category_id + 1,
                        name=category_name,
                    )
                ]
            )

    for image_id, image_path, label_path in zip(
        image_ids,
        search.get_image_files(image_dir),
        search.get_files(label_dir, "txt"),
    ):
        # image
        img = cv2.imread(str(image_path))
        height, width, _ = img.shape
        image = Image.new(
            image_id=image_id,
            file_name=f"{image_id}{image_path.suffix}",
            width=width,
            height=height,
        )
        self.add_images([image])

        # annotation
        with open(label_path) as f:  # TODO: use load_txt of waffle_utils after implementing
            txt = f.readlines()

        current_annotation_id = len(self.get_annotations())
        for i, t in enumerate(txt, start=1):
            category_id, x, y, w, h = list(map(float, t.split()))
            category_id = int(category_id) + 1
            x *= width
            y *= height
            w *= width
            h *= height

            x -= w / 2
            y -= h / 2

            x, y, w, h = int(x), int(y), int(w), int(h)
            annotation = Annotation.object_detection(
                annotation_id=current_annotation_id + i,
                image_id=image_id,
                category_id=category_id,
                bbox=[x, y, w, h],
                area=w * h,
            )
            self.add_annotations([annotation])

        # raw
        dst = self.raw_image_dir / f"{image_id}{image_path.suffix}"
        io.copy_file(image_path, dst)


def import_yolo(self, yaml_path: str):
    if self.task == TaskType.OBJECT_DETECTION:
        _import = _import_yolo_object_detection
    elif self.task == TaskType.CLASSIFICATION:
        _import = _import_yolo_classification
    else:
        raise ValueError(f"Unsupported task: {self.task}")

    info = io.load_yaml(yaml_path)
    yolo_root_dir = Path(info["path"])

    current_image_id = 1
    for set_type in ["train", "val", "test"]:
        if set_type not in info.keys():
            continue

        # sets
        set_dir = Path(yolo_root_dir) / set_type
        image_num = len(search.get_image_files(set_dir))
        image_ids = list(range(current_image_id, image_num + current_image_id))

        io.save_json(image_ids, self.set_dir / f"{set_type}.json", True)
        current_image_id += image_num

        # import other field
        _import(self, set_dir, image_ids, info)

    # TODO: add unlabeled set
    io.save_json([], self.unlabeled_set_file, create_directory=True)

    return self.dataset_dir
