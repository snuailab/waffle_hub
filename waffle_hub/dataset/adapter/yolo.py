import warnings
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from waffle_utils.file import io, search

from waffle_hub import TaskType
from waffle_hub.schema.data import ImageInfo
from waffle_hub.schema.fields import Annotation, Category, Image
from waffle_hub.temp_utils.image.io import load_image
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


def resize_image(
    image: np.ndarray, image_size: list[int], letter_box: bool = False
) -> list[np.ndarray, ImageInfo]:
    """Resize Image.

    Args:
        image (np.ndarray): opencv image.
        image_size (list[int]): image [width, height].
        letter_box (bool): letter box.

    Returns:
        list[np.ndarray, ImageInfo]: resized image, image info.
    """

    h, w = image.shape[:2]
    W, H = image_size

    if letter_box:
        h_ratio = H / h
        w_ratio = W / w

        if w_ratio < h_ratio:
            resize_shape = (int(w * w_ratio), round(h * w_ratio))
            total_pad = H - resize_shape[1]
            top = total_pad // 2
            bottom = total_pad - top
            left, right = 0, 0
        else:
            resize_shape = (round(w * h_ratio), int(h * h_ratio))
            total_pad = W - resize_shape[0]
            left = total_pad // 2
            right = total_pad - left
            top, bottom = 0, 0

        resized_image = cv2.resize(image, resize_shape, interpolation=cv2.INTER_LINEAR)
        resized_image = cv2.copyMakeBorder(
            resized_image, top, bottom, left, right, None, value=(114, 114, 114)
        )

    else:
        resize_shape = (W, H)
        left, top = 0, 0
        resized_image = cv2.resize(image, resize_shape, interpolation=cv2.INTER_LINEAR)

    return resized_image, ImageInfo(
        ori_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        ori_shape=(w, h),
        new_shape=resize_shape,
        input_shape=(W, H),
        pad=(left, top),
    )


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
            # TODO : resize_image를 dataset.utils로 분할, make_directory를 사용하지 않고 dataset.utils에 image save하는 경우 create_directory=True를 할 수 있도록
            io.make_directory(image_dst_path.parent)
            img = cv2.imread(str(image_path))
            transformed_image, image_info = resize_image(img, image_size=[224, 224], letter_box=True)
            cv2.imwrite(str(image_dst_path), transformed_image)


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
            if annotations == []:  # for background image
                continue

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
            if annotations == []:  # for background image
                continue

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
    background_ids = self.get_background_ids()

    if self.task == TaskType.CLASSIFICATION:
        _export_yolo_classification(self, export_dir, train_ids, val_ids, test_ids, [])
    elif self.task == TaskType.OBJECT_DETECTION:
        train_ids += background_ids
        _export_yolo_detection(self, export_dir, train_ids, val_ids, test_ids, [])
    elif self.task == TaskType.INSTANCE_SEGMENTATION:
        train_ids += background_ids
        _export_yolo_segmentation(self, export_dir, train_ids, val_ids, test_ids, [])
    else:
        raise ValueError(f"Unsupported task type: {self.task}")

    io.save_yaml(
        {
            "path": str(export_dir.absolute()),
            "train": "train",
            "val": "val",
            "test": "test" if test_ids else "val",
            "names": {category.category_id - 1: category.name for category in self.get_categories()},
        },
        export_dir / "data.yaml",
    )

    return str(export_dir)


# import functions
def _get_yolo_image_rel_paths(yolo_root_dir: Path, set_paths: list = ["train", "val", "test"]):
    set_image_rel_path = {}
    for set_path in set_paths:
        set_image_rel_path[set_path] = list(
            map(
                lambda x: x.relative_to(yolo_root_dir),
                search.get_image_files(yolo_root_dir / set_path),
            )
        )
    return set_image_rel_path


def _import_yolo_classification(self, yolo_root_dir: Path, *args):
    # get image relative paths
    set_image_rel_path = _get_yolo_image_rel_paths(yolo_root_dir, ["train", "val", "test"])

    # get categories
    category_group = groupby(set_image_rel_path["train"], lambda image_path: image_path.parts[1])
    category_names = {k for k, _ in category_group}
    category_name2id = {}
    for category_id, category_name in enumerate(category_names, start=1):
        category_name2id[category_name] = category_id
        self.add_categories(
            [
                Category.classification(
                    category_id=category_id,
                    name=category_name,
                )
            ]
        )

    # convert to waffle format
    image_id = 1
    annotation_id = 1
    for set_name, image_rel_paths in set_image_rel_path.items():
        set_image_ids = []
        for image_rel_path in image_rel_paths:
            image_path = yolo_root_dir / image_rel_path
            file_name = f"{image_id}{image_path.suffix}"

            height, width = load_image(image_path).shape[:2]
            self.add_images(
                [
                    Image.new(
                        image_id=image_id,
                        file_name=file_name,
                        width=width,
                        height=height,
                        original_file_name=image_rel_path,
                    )
                ]
            )
            set_image_ids.append(image_id)
            io.copy_file(image_path, self.raw_image_dir / file_name, create_directory=True)

            category_name = image_rel_path.parts[
                1
            ]  # image rel path: {set_name}/{category_name}/{file_name}
            self.add_annotations(
                [
                    Annotation.classification(
                        annotation_id=annotation_id,
                        image_id=image_id,
                        category_id=category_name2id[category_name],
                    )
                ]
            )

            image_id += 1
            annotation_id += 1

        io.save_json(set_image_ids, self.set_dir / f"{set_name}.json", True)

    return True


def _parse_od_label(line: str, width: int, height: int):
    """parse label file for od"""
    category_id, cx, cy, w, h = map(float, line.split())
    x1 = cx - w / 2
    y1 = cy - h / 2
    return {
        "category_id": int(category_id) + 1,
        "bbox": [x1 * width, y1 * height, w * width, h * height],
    }


def _parse_seg_label(line: str, width: int, height: int):
    """parse label file for seg"""
    category_id, *segment = map(float, line.split())
    segment[::2] = [int(x * width) for x in segment[::2]]
    segment[1::2] = [y * height for y in segment[1::2]]
    return {
        "category_id": int(category_id) + 1,
        "segmentation": [segment],
    }


def _import_yolo_images_labels(self, yolo_root_dir: Path, yaml_path: str, task: TaskType):
    """import function for od, seg, keypoint"""

    parse_func = {
        TaskType.OBJECT_DETECTION: _parse_od_label,
        TaskType.INSTANCE_SEGMENTATION: _parse_seg_label,
    }[task]

    info = io.load_yaml(yaml_path)
    # get image relative paths
    set_image_rel_path = _get_yolo_image_rel_paths(
        yolo_root_dir,
        list(
            set(
                map(
                    lambda x: info.get(x, None),
                    [info["train"], info["val"], info.get("test", info["val"])],
                )
            )
        ),
    )

    # get categories
    names = info["names"]
    category_name2id = {}
    if isinstance(names, list):
        names = {category_id: category_name for category_id, category_name in enumerate(names)}
    for category_id, category_name in names.items():
        category_name2id[category_name] = category_id + 1
        self.add_categories(
            [
                Category.new(
                    category_id=category_id + 1,
                    name=category_name,
                    task=task,
                )
            ]
        )

    # convert to waffle format
    image_id = 1
    annotation_id = 1
    for set_name, image_rel_paths in set_image_rel_path.items():
        set_image_ids = []
        for image_rel_path in image_rel_paths:
            image_path = (
                yolo_root_dir / image_rel_path
            )  # image rel path: {set_name}/images/{file_name(.EXT)}
            file_name = f"{image_id}{image_path.suffix}"

            height, width = load_image(image_path).shape[:2]
            self.add_images(
                [
                    Image.new(
                        image_id=image_id,
                        file_name=file_name,
                        width=width,
                        height=height,
                        original_file_name=image_rel_path,
                    )
                ]
            )
            set_image_ids.append(image_id)
            io.copy_file(image_path, self.raw_image_dir / file_name, create_directory=True)

            label_path = image_rel_path.with_suffix(
                ".txt"
            )  # label rel path: {set_name}/labels/{file_name(.txt)}
            label_parts = list(label_path.parts)
            label_parts[1] = "labels"
            label_path = yolo_root_dir / Path(*label_parts)
            with label_path.open("r") as f:
                lines = f.readlines()

            for line in lines:
                annotation = parse_func(line, width, height)
                self.add_annotations(
                    [
                        Annotation.new(
                            annotation_id=annotation_id,
                            image_id=image_id,
                            task=task,
                            **annotation,
                        )
                    ]
                )
                annotation_id += 1
            image_id += 1

        io.save_json(set_image_ids, self.set_dir / f"{set_name}.json", True)


def import_yolo(self, yolo_root_dir: str, yaml_path: str):
    """
    Import YOLO dataset.

    Args:
        yaml_path (str): Path to the yaml file.
    """
    if self.task == TaskType.OBJECT_DETECTION:
        _import = _import_yolo_images_labels
    elif self.task == TaskType.CLASSIFICATION:
        _import = _import_yolo_classification
    elif self.task == TaskType.INSTANCE_SEGMENTATION:
        _import = _import_yolo_images_labels
    else:
        raise ValueError(f"Unsupported task: {self.task}")

    _import(self, Path(yolo_root_dir), yaml_path, self.task)

    # TODO: add unlabeled set
    io.save_json([], self.unlabeled_set_file, create_directory=True)
