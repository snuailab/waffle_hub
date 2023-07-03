import logging
import warnings
from pathlib import Path
from typing import Union

import tqdm
from pycocotools.coco import COCO
from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.schema.fields import Annotation, Category, Image
from waffle_hub.utils.conversion import convert_rle_to_polygon


def _export_coco(
    self,
    export_dir: Path,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    unlabeled_ids: list,
):
    """Export dataset to COCO format

    Args:
        export_dir (Path): Path to export directory
        train_ids (list): List of train ids
        val_ids (list): List of validation ids
        test_ids (list): List of test ids
        unlabeled_ids (list): List of unlabeled ids
    """
    io.make_directory(export_dir)

    image_dir = export_dir / "images"

    for split, image_ids in zip(
        ["train", "val", "test", "unlabeled"],
        [train_ids, val_ids, test_ids, unlabeled_ids],
    ):
        if len(image_ids) == 0:
            continue

        coco = {
            "categories": [
                {
                    "id": category.category_id,
                    "name": category.name,
                    "supercategory": category.supercategory,
                }
                for category in self.get_categories()
            ],
            "images": [],
            "annotations": [],
        }

        for image in self.get_images(image_ids):
            image_path = self.raw_image_dir / image.file_name
            image_dst_path = image_dir / image.file_name
            io.copy_file(image_path, image_dst_path, create_directory=True)

            d = image.to_dict()
            image_id = d.pop("image_id")
            coco["images"].append({"id": image_id, **d})

            annotations = self.get_annotations(image_id)
            for annotation in annotations:
                d = annotation.to_dict()
                if d.get("segmentation", None):
                    if isinstance(d["segmentation"], dict):
                        d["segmentation"] = convert_rle_to_polygon(d["segmentation"])
                annotation_id = d.pop("annotation_id")
                coco["annotations"].append({"id": annotation_id, **d})

        io.save_json(coco, export_dir / f"{split}.json", create_directory=True)


def export_coco(self, export_dir: Union[str, Path]) -> str:
    """Export dataset to COCO format

    Args:
        export_dir (str): Path to export directory

    Returns:
        str: Path to export directory
    """
    export_dir = Path(export_dir)

    train_ids, val_ids, test_ids, _ = self.get_split_ids()

    if self.task == TaskType.CLASSIFICATION:
        _export_coco(self, export_dir, train_ids, val_ids, test_ids, [])
    elif self.task == TaskType.OBJECT_DETECTION:
        _export_coco(self, export_dir, train_ids, val_ids, test_ids, [])
    elif self.task == TaskType.INSTANCE_SEGMENTATION:
        _export_coco(self, export_dir, train_ids, val_ids, test_ids, [])
    else:
        raise ValueError(f"Unsupported task type: {self.task}")

    return str(export_dir)


def import_coco(self, coco_files: list[str], coco_root_dirs: list[str]):
    """
    Import coco dataset

    Args:
        coco_files (list[str]): List of coco annotation files
        coco_root_dirs (list[str]): List of coco root directories
    """
    if len(coco_files) == 1:
        set_names = [None]
    elif len(coco_files) == 2:
        set_names = ["train", "val"]
    elif len(coco_files) == 3:
        set_names = ["train", "val", "test"]
    else:
        raise ValueError("coco_file should have 1, 2, or 3 files.")

    cocos = [COCO(coco_file) for coco_file in coco_files]

    # categories should be same between coco files
    categories = cocos[0].loadCats(cocos[0].getCatIds())
    for coco in cocos[1:]:
        if categories != coco.loadCats(coco.getCatIds()):
            raise ValueError("categories should be same between coco files.")

    coco_cat_id_to_waffle_cat_id = {}
    for i, category in enumerate(categories, start=1):
        coco_category_id = category.pop("id")
        coco_cat_id_to_waffle_cat_id[coco_category_id] = i
        self.add_categories([Category.from_dict({**category, "category_id": i}, task=self.task)])

    # import coco dataset
    total_length = sum([len(coco.getImgIds()) for coco in cocos])
    logging.info(f"Importing coco dataset. Total length: {total_length}")
    pgbar = tqdm.tqdm(total=total_length, desc="Importing coco dataset")

    image_id = 1
    annotation_id = 1

    # parse coco annotation file
    for coco, coco_root_dir, set_name in tqdm.tqdm(zip(cocos, coco_root_dirs, set_names)):

        image_ids = []
        for coco_image_id, annotation_dicts in coco.imgToAnns.items():
            if len(annotation_dicts) == 0:
                warnings.warn(f"image_id {coco_image_id} has no annotations.")
                continue

            image_dict = coco.loadImgs(coco_image_id)[0]
            image_dict.pop("id")

            file_name = image_dict.pop("file_name")
            image_path = Path(coco_root_dir) / file_name
            if not image_path.exists():
                raise FileNotFoundError(f"{image_path} does not exist.")

            if set_name:
                file_name = f"{set_name}/{file_name}"

            self.add_images(
                [Image.from_dict({**image_dict, "image_id": image_id, "file_name": file_name})]
            )
            io.copy_file(image_path, self.raw_image_dir / file_name, create_directory=True)

            for annotation_dict in annotation_dicts:
                annotation_dict.pop("id")
                self.add_annotations(
                    [
                        Annotation.from_dict(
                            {
                                **annotation_dict,
                                "image_id": image_id,
                                "annotation_id": annotation_id,
                                "category_id": coco_cat_id_to_waffle_cat_id[
                                    annotation_dict["category_id"]
                                ],
                            },
                            task=self.task,
                        )
                    ]
                )
                annotation_id += 1

            image_ids.append(image_id)
            image_id += 1
            pgbar.update(1)

        if set_name:
            io.save_json(image_ids, self.set_dir / f"{set_name}.json", create_directory=True)

    pgbar.close()
