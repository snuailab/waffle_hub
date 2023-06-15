from pathlib import Path
from typing import Union

from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.utils.conversion import convert_rle_to_polygon


def _export_autocare_dlt(
    self,
    export_dir: Path,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    unlabeled_ids: list,
):
    """Export dataset to Autocare DLT format

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

            for annotation in self.get_annotations(image_id):
                d = annotation.to_dict()
                if d.get("segmentation", None):
                    if isinstance(d["segmentation"], dict):
                        d["segmentation"] = convert_rle_to_polygon(d["segmentation"])
                if d.get("caption", None) and (not d.get("category_id", None)):
                    d["category_id"] = 1  # dummy for ocr
                annotation_id = d.pop("annotation_id")
                coco["annotations"].append({"id": annotation_id, **d})

        io.save_json(coco, export_dir / f"{split}.json", create_directory=True)


def export_autocare_dlt(self, export_dir: Union[str, Path]) -> str:
    """Export dataset to Autocare DLT format

    Args:
        export_dir (Union[str, Path]): Path to export directory

    Returns:
        str: Path to export directory
    """
    export_dir = Path(export_dir)

    train_ids, val_ids, test_ids, unlabeled_ids = self.get_split_ids()

    if self.task == TaskType.CLASSIFICATION:
        _export_autocare_dlt(self, export_dir, train_ids, val_ids, test_ids, [])
    elif self.task == TaskType.OBJECT_DETECTION:
        _export_autocare_dlt(self, export_dir, train_ids, val_ids, test_ids, [])
    elif self.task == TaskType.INSTANCE_SEGMENTATION:
        _export_autocare_dlt(self, export_dir, train_ids, val_ids, test_ids, [])
    elif self.task == TaskType.TEXT_RECOGNITION:
        _export_autocare_dlt(self, export_dir, train_ids, val_ids, test_ids, [])
    else:
        raise ValueError(f"Unsupported task type: {self.task}")

    return str(export_dir)
