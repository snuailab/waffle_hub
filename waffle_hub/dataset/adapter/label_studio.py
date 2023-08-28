from pathlib import Path

import cv2
from tqdm import tqdm
from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.schema.fields import Annotation, Category, Image


def import_object_detection(self, json_file, image_dir=None):

    images = []
    annotations = []
    categories = []
    category_to_id = {}

    datas = io.load_json(json_file)

    for image_id, data in tqdm(enumerate(datas, start=1), total=len(datas)):

        image_file_name = data["file_upload"]
        src_image_path = Path(image_dir) / image_file_name if image_dir else data["data"]["image"]
        io.copy_file(src_image_path, self.raw_image_dir / image_file_name, create_directory=True)

        H, W = cv2.imread(str(src_image_path)).shape[:2]

        image = Image.new(
            image_id=image_id,
            file_name=str(image_file_name),
            width=W,
            height=H,
        )
        images.append(image)

        for annotation in data["annotations"]:
            category = annotation["result"][0]["value"]["rectanglelabels"][0]

            if category not in category_to_id:
                category_to_id[category] = len(category_to_id) + 1
                categories.append(
                    Category.object_detection(
                        category_id=category_to_id[category],
                        name=category,
                        supercategory="object",
                    )
                )

            x = annotation["result"][0]["value"]["x"] * W / 100
            y = annotation["result"][0]["value"]["y"] * H / 100
            width = annotation["result"][0]["value"]["width"] * W / 100
            height = annotation["result"][0]["value"]["height"] * H / 100

            annotations.append(
                Annotation.object_detection(
                    annotation_id=len(annotations) + 1,
                    image_id=image_id,
                    category_id=category_to_id[category],
                    bbox=[x, y, width, height],
                    is_crowd=False,
                )
            )

    self.add_images(images)
    self.add_categories(categories)
    self.add_annotations(annotations)


def import_classification(self, json_file, image_dir):

    images = []
    annotations = []
    categories = []
    category_to_id = {}

    datas = io.load_json(json_file)

    for image_id, data in tqdm(enumerate(datas, start=1), total=len(datas)):

        image_file_name = data["file_upload"]
        src_image_path = Path(image_dir) / image_file_name if image_dir else data["data"]["image"]
        io.copy_file(src_image_path, self.raw_image_dir / image_file_name, create_directory=True)

        H, W = cv2.imread(str(src_image_path)).shape[:2]

        image = Image.new(
            image_id=image_id,
            file_name=str(image_file_name),
            width=W,
            height=H,
        )
        images.append(image)

        for annotation in data["annotations"]:
            category = annotation["result"][0]["value"]["choices"][0]

            if category not in category_to_id:
                category_to_id[category] = len(category_to_id) + 1
                categories.append(
                    Category.classification(
                        category_id=category_to_id[category],
                        name=category,
                    )
                )

            annotations.append(
                Annotation.classification(
                    annotation_id=len(annotations) + 1,
                    image_id=image_id,
                    category_id=category_to_id[category],
                )
            )

    self.add_images(images)
    self.add_categories(categories)
    self.add_annotations(annotations)


def import_label_studio(self, json_file, task, image_dir=None):

    if task == TaskType.OBJECT_DETECTION:
        _import = import_object_detection
    elif task == TaskType.CLASSIFICATION:
        _import = import_classification
    else:
        raise NotImplementedError(f"Task type {task} not implemented")

    _import(self, json_file, image_dir)

    # TODO: add unlabeled set
    io.save_json([], self.unlabeled_set_file, create_directory=True)
