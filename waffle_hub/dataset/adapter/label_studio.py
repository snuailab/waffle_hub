from pathlib import Path

from tqdm import tqdm
from waffle_utils.file import io
from waffle_utils.image.io import load_image

from waffle_hub import TaskType
from waffle_hub.schema.fields import Annotation, Category, Image


def import_object_detection(self, json_file, image_dir=None):
    """
    Label studio object detection format

    [
        {
            "id": 1,
            "data": {
                "image": "/data/upload/1/1.jpg"  # {data_path}/{project_id}/{file_name}
            },
            "file_upload": "1.jpg",  # {file_name}
            "annotations": [
                {
                    "id": 1,
                    "result": [
                        {
                            "id": 1,
                            "type": "rectanglelabels",
                            "value": {
                                "rectanglelabels": ["cat"],
                                "x": 20.3,  # left  # 0 ~ 100 (%)
                                "y": 34.2,  # top  # 0 ~ 100 (%)
                                "width": 10.4,  # width  # 0 ~ 100 (%)
                                "height": 10.2  # height  # 0 ~ 100 (%)
                            },
                            ...
                        },
                        ...
                    ],
                    ...
                },
                ...
            ],
            ...
        },
        ...
    ]
    """

    images = []
    annotations = []
    categories = []
    category_to_id = {}

    datas = io.load_json(json_file)

    for image_id, data in tqdm(enumerate(datas, start=1), total=len(datas)):

        image_file_name = data["file_upload"]
        src_image_path = Path(image_dir) / image_file_name if image_dir else data["data"]["image"]
        io.copy_file(src_image_path, self.raw_image_dir / image_file_name, create_directory=True)

        H, W = load_image(src_image_path).shape[:2]

        image = Image.new(
            image_id=image_id,
            file_name=str(image_file_name),
            width=W,
            height=H,
        )
        images.append(image)

        for annotation in data["annotations"]:
            for result in annotation["result"]:
                if "rectanglelabels" not in result["value"]:
                    continue
                value = result["value"]

                category = value["rectanglelabels"][0]
                if category not in category_to_id:
                    category_to_id[category] = len(category_to_id) + 1
                    categories.append(
                        Category.object_detection(
                            category_id=category_to_id[category],
                            name=category,
                            supercategory="object",
                        )
                    )

                x = value["x"] * W / 100
                y = value["y"] * H / 100
                width = value["width"] * W / 100
                height = value["height"] * H / 100

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
    """
    Label studio classification format

    [
        {
            "id": 1,
            "data": {
                "image": "/data/upload/1/1.jpg"  # {data_path}/{project_id}/{file_name}
            },
            "file_upload": "1.jpg",  # {file_name}
            "annotations": [
                {
                    "id": 1,
                    "result": [
                        {
                            "id": 1,
                            "type": "choices",
                            "value": {
                                "choices": ["cat"]
                            },
                            ...
                        },
                        ...
                    ],
                    ...
                },
                ...
            ],
            ...
        },
        ...
    ]
    """

    images = []
    annotations = []
    categories = []
    category_to_id = {}

    datas = io.load_json(json_file)

    for image_id, data in tqdm(enumerate(datas, start=1), total=len(datas)):

        image_file_name = data["file_upload"]
        src_image_path = Path(image_dir) / image_file_name if image_dir else data["data"]["image"]
        io.copy_file(src_image_path, self.raw_image_dir / image_file_name, create_directory=True)

        H, W = load_image(src_image_path).shape[:2]

        image = Image.new(
            image_id=image_id,
            file_name=str(image_file_name),
            width=W,
            height=H,
        )
        images.append(image)

        for annotation in data["annotations"]:
            for result in annotation["result"]:
                if "choices" not in result["value"]:
                    continue
                value = result["value"]

                category = value["choices"][0]
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
