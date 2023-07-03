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


def _import_yolo_classification(self, yolo_root_dir: Path, *args):
    # categories
    image_paths = set()
    category_names = set()
    for set_type in ["train", "val", "test"]:
        set_dir = yolo_root_dir / set_type
        for category_name in [path for path in set_dir.glob("*") if path.is_dir()]:
            category_name = _remove_root_dir(set_dir, category_name).parts[0]
            category_names.add(category_name)
            image_paths |= {
                _remove_root_dir(set_dir / category_name, image_path)
                for image_path in search.get_image_files(set_dir / category_name)
            }
    image_path2id = {image_path: i for i, image_path in enumerate(image_paths, start=1)}

    category_name2id = {}
    for category_id, category_name in enumerate(sorted(list(category_names)), start=1):
        category_name2id[category_name] = category_id
        self.add_categories(
            [
                Category.classification(
                    category_id=category_id,
                    name=category_name,
                )
            ]
        )

    for set_type in ["train", "val", "test"]:
        image_ids = []
        for category_name in category_names:
            set_dir = yolo_root_dir / set_type / category_name
            category_id = category_name2id[category_name]
            for image_path in search.get_image_files(set_dir):
                file_name = _remove_root_dir(set_dir, image_path)
                image_id = image_path2id[file_name]
                image_ids.append(image_id)

                # image
                img = cv2.imread(str(image_path))
                height, width, _ = img.shape
                image = Image.new(
                    image_id=image_id,
                    file_name=str(file_name),
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
                dst = self.raw_image_dir / file_name
                io.copy_file(image_path, dst, True)

        io.save_json(image_ids, self.set_dir / f"{set_type}.json", True)


def _import_yolo_object_detection(self, yolo_root_dir: Path, yaml_path: str):
    # categories
    info = io.load_yaml(yaml_path)
    names = info["names"]
    if isinstance(names, list):
        names = {category_id: category_name for category_id, category_name in enumerate(names)}
    for category_id, category_name in names.items():
        self.add_categories(
            [
                Category.object_detection(
                    category_id=category_id + 1,
                    name=category_name,
                )
            ]
        )

    image_paths = set()
    for set_type in ["train", "val", "test"]:
        set_dir = yolo_root_dir / set_type / "images"
        image_paths |= {
            _remove_root_dir(set_dir, image_path) for image_path in search.get_image_files(set_dir)
        }
    image_path2id = {image_path: i for i, image_path in enumerate(image_paths, start=1)}

    for set_type in ["train", "val", "test"]:
        set_dir = yolo_root_dir / set_type
        image_dir = set_dir / "images"
        label_dir = set_dir / "labels"

        if not image_dir.exists():
            raise FileNotFoundError(f"{image_dir} does not exist.")
        if not label_dir.exists():
            raise FileNotFoundError(f"{label_dir} does not exist.")

        image_ids = []
        for image_path in search.get_image_files(image_dir):
            # image
            file_name = _remove_root_dir(image_dir, image_path)
            image_id = image_path2id[file_name]
            image_ids.append(image_id)
            img = cv2.imread(str(image_path))
            height, width, _ = img.shape
            image = Image.new(
                image_id=image_id,
                file_name=str(file_name),
                width=width,
                height=height,
            )
            self.add_images([image])

            # annotation
            label_path = label_dir / file_name.with_suffix(".txt")
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
            dst = self.raw_image_dir / file_name
            io.copy_file(image_path, dst, True)
        io.save_json(image_ids, self.set_dir / f"{set_type}.json", True)


def _import_yolo_instance_segmentation(self, yolo_root_dir: Path, yaml_path: str):
    # categories
    info = io.load_yaml(yaml_path)
    names = info["names"]
    if isinstance(names, list):
        names = {category_id: category_name for category_id, category_name in enumerate(names)}
    for category_id, category_name in names.items():
        self.add_categories(
            [
                Category.object_detection(
                    category_id=category_id + 1,
                    name=category_name,
                )
            ]
        )

    image_paths = set()
    for set_type in ["train", "val", "test"]:
        set_dir = yolo_root_dir / set_type / "images"
        image_paths |= {
            _remove_root_dir(set_dir, image_path) for image_path in search.get_image_files(set_dir)
        }
    image_path2id = {image_path: i for i, image_path in enumerate(image_paths, start=1)}

    for set_type in ["train", "val", "test"]:
        set_dir = yolo_root_dir / set_type
        image_dir = set_dir / "images"
        label_dir = set_dir / "labels"

        if not image_dir.exists():
            raise FileNotFoundError(f"{image_dir} does not exist.")
        if not label_dir.exists():
            raise FileNotFoundError(f"{label_dir} does not exist.")

        image_ids = []
        for image_path in search.get_image_files(image_dir):
            # image
            file_name = _remove_root_dir(image_dir, image_path)
            image_id = image_path2id[file_name]
            image_ids.append(image_id)
            img = cv2.imread(str(image_path))
            height, width, _ = img.shape
            image = Image.new(
                image_id=image_id,
                file_name=str(file_name),
                width=width,
                height=height,
            )
            self.add_images([image])

            # annotation
            label_path = label_dir / file_name.with_suffix(".txt")
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
                annotation = Annotation.instance_segmentation(
                    annotation_id=current_annotation_id + i,
                    image_id=image_id,
                    category_id=category_id,
                    segmentation=[],
                    bbox=[x, y, w, h],
                    area=w * h,
                )
                self.add_annotations([annotation])

            # raw
            dst = self.raw_image_dir / file_name
            io.copy_file(image_path, dst, True)
        io.save_json(image_ids, self.set_dir / f"{set_type}.json", True)


def import_yolo(self, yolo_root_dir: str, yaml_path: str):
    """
    Import YOLO dataset.

    Args:
        yaml_path (str): Path to the yaml file.
    """
    if self.task == TaskType.OBJECT_DETECTION:
        _import = _import_yolo_object_detection
    elif self.task == TaskType.CLASSIFICATION:
        _import = _import_yolo_classification
    elif self.task == TaskType.INSTANCE_SEGMENTATION:
        _import = _import_yolo_instance_segmentation
    else:
        raise ValueError(f"Unsupported task: {self.task}")

    if isinstance(yolo_root_dir, str):
        yolo_root_dir = Path(yolo_root_dir)

    _import(self, yolo_root_dir, yaml_path)

    # TODO: add unlabeled set
    io.save_json([], self.unlabeled_set_file, create_directory=True)


def _remove_root_dir(root_dir: Path, target_dir: Path) -> Path:
    """
    Remove root dir from target dir

    Args:
        root_dir (Path): Root directory
        target_dir (Path): Target directory

    Returns:
        Path: Relative path from root dir to target dir
    """
    idx = 0
    for root_part, target_part in zip(root_dir.parts, target_dir.parts):
        if root_part != target_part:
            break
        idx += 1

    return Path(*target_dir.parts[idx:])
