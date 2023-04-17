import logging
import random
import warnings
from collections import OrderedDict
from functools import cached_property
from pathlib import Path
from typing import Union

from waffle_utils.file import io
from waffle_utils.utils import type_validator

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from waffle_hub import DataType, TaskType
from waffle_hub.dataset.adapter import (
    export_coco,
    export_huggingface,
    export_yolo,
)
from waffle_hub.schema import Annotation, Category, DatasetInfo, Image

logger = logging.getLogger(__name__)


class Dataset:
    DEFAULT_DATASET_ROOT_DIR = Path("./datasets")
    DATASET_INFO_FILE_NAME = Path("info.yaml")

    RAW_IMAGE_DIR = Path("raw")
    IMAGE_DIR = Path("images")
    ANNOTATION_DIR = Path("annotations")
    CATEGORY_DIR = Path("categories")
    PREDICTION_DIR = Path("predictions")
    EXPORT_DIR = Path("exports")
    SET_DIR = Path("sets")

    TRAIN_SET_FILE_NAME = Path("train.json")
    VAL_SET_FILE_NAME = Path("val.json")
    TEST_SET_FILE_NAME = Path("test.json")
    UNLABELED_SET_FILE_NAME = Path("unlabeled.json")

    def __init__(
        self,
        name: str,
        task: Union[str, TaskType],
        root_dir: str = None,
    ):
        self.name = name
        self.task = task
        self.root_dir = (
            Path(root_dir) if root_dir else Dataset.DEFAULT_DATASET_ROOT_DIR
        )

    # properties
    @property
    def name(self):
        return self.__name

    @name.setter
    @type_validator(str)
    def name(self, v):
        self.__name = v

    @property
    def task(self):
        return self.__task

    @task.setter
    def task(self, v):
        if v not in TaskType:
            raise ValueError(
                f"Invalid task type: {v}"
                f"Available task types: {list(TaskType)}"
            )
        self.__task = str(v).upper()

    @property
    def root_dir(self):
        return self.__root_dir

    @root_dir.setter
    @type_validator(Path)
    def root_dir(self, v):
        self.__root_dir = v

    # cached properties
    @cached_property
    def dataset_dir(self) -> Path:
        return self.root_dir / self.name

    @cached_property
    def dataset_info_file(self) -> Path:
        return self.dataset_dir / Dataset.DATASET_INFO_FILE_NAME

    @cached_property
    def raw_image_dir(self) -> Path:
        return self.dataset_dir / Dataset.RAW_IMAGE_DIR

    @cached_property
    def image_dir(self) -> Path:
        return self.dataset_dir / Dataset.IMAGE_DIR

    @cached_property
    def annotation_dir(self) -> Path:
        return self.dataset_dir / Dataset.ANNOTATION_DIR

    @cached_property
    def prediction_dir(self) -> Path:
        return self.dataset_dir / Dataset.PREDICTION_DIR

    @cached_property
    def category_dir(self) -> Path:
        return self.dataset_dir / Dataset.CATEGORY_DIR

    @cached_property
    def export_dir(self) -> Path:
        return self.dataset_dir / Dataset.EXPORT_DIR

    @cached_property
    def set_dir(self) -> Path:
        return self.dataset_dir / Dataset.SET_DIR

    @cached_property
    def train_set_file(self) -> Path:
        return self.set_dir / Dataset.TRAIN_SET_FILE_NAME

    @cached_property
    def val_set_file(self) -> Path:
        return self.set_dir / Dataset.VAL_SET_FILE_NAME

    @cached_property
    def test_set_file(self) -> Path:
        return self.set_dir / Dataset.TEST_SET_FILE_NAME

    @cached_property
    def unlabeled_set_file(self) -> Path:
        return self.set_dir / Dataset.UNLABELED_SET_FILE_NAME

    @cached_property
    def images(self) -> dict[int, Image]:
        return OrderedDict(
            {
                image.image_id: image
                for image in sorted(
                    self.get_images(), key=lambda i: i.image_id
                )
            }
        )

    @cached_property
    def unlabeled_images(self) -> dict[int, Image]:
        return OrderedDict(
            {
                image.image_id: image
                for image in sorted(
                    self.get_images(labeled=False), key=lambda i: i.image_id
                )
            }
        )

    @cached_property
    def annotations(self) -> dict[int, Annotation]:
        return OrderedDict(
            {
                annotation.annotation_id: annotation
                for annotation in sorted(
                    self.get_annotations(), key=lambda a: a.annotation_id
                )
            }
        )

    @cached_property
    def categories(self) -> dict[int, Category]:
        return OrderedDict(
            {
                category.category_id: category
                for category in sorted(
                    self.get_categories(), key=lambda c: c.category_id
                )
            }
        )

    @cached_property
    def category_names(self) -> list[str]:
        categories: dict[int, Category] = self.categories
        return [
            c.name
            for c in sorted(categories.values(), key=lambda c: c.category_id)
        ]

    @cached_property
    def image_to_annotations(self) -> dict[int, list[Annotation]]:
        return OrderedDict(
            {
                image_id: list(
                    filter(
                        lambda a: a.image_id == image_id,
                        self.annotations.values(),
                    )
                )
                for image_id in self.images.keys()
            }
        )

    # factories
    @classmethod
    def new(cls, name: str, task: str, root_dir: str = None) -> "Dataset":
        """Create New Dataset

        Args:
            name (str): Dataset name
            task (str): Dataset task
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Raises:
            FileExistsError: if dataset name already exists

        Returns:
            Dataset: Dataset Class
        """
        ds = cls(name, task, root_dir)
        if ds.initialized():
            raise FileExistsError(
                f'{ds.dataset_dir} already exists. try another name or Dataset.load("{name}")'
            )
        ds.initialize()
        return ds

    @classmethod
    def clone(
        cls,
        src_name: str,
        name: str,
        src_root_dir: str = None,
        root_dir: str = None,
    ) -> "Dataset":
        """Clone Existing Dataset

        Args:
            src_name (str):
                Dataset name to clone.
                It should be Waffle Created Dataset.
            name (str): New Dataset name
            src_root_dir (str, optional): Source Dataset root directory. Defaults to None.
            root_dir (str, optional): New Dataset root directory. Defaults to None.

        Raises:
            FileNotFoundError: if source dataset does not exist.
            FileExistsError: if new dataset name already exist.

        Returns:
            Dataset: Dataset Class
        """
        src_ds = Dataset.load(src_name, src_root_dir)
        if not src_ds.initialized():
            raise FileNotFoundError(
                f"{src_ds.dataset_dir} has not been created by Waffle."
            )

        ds = Dataset.new(name, src_ds.task, root_dir)
        ds.initialize()
        io.copy_files_to_directory(
            src_ds.dataset_dir, ds.dataset_dir, create_directory=True
        )

        return ds

    @classmethod
    def load(cls, name: str, root_dir: str = None) -> "Dataset":
        """Load Dataset.

        Args:
            name (str): Dataset name that Waffle Created
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Raises:
            FileNotFoundError: if source dataset does not exist.

        Returns:
            Dataset: Dataset Class
        """
        dataset_info_file = (
            Path(root_dir) / name / Dataset.DATASET_INFO_FILE_NAME
        )
        if not dataset_info_file.exists():
            raise FileNotFoundError(
                f"{dataset_info_file} has not been created."
            )
        dataset_info = DatasetInfo.from_yaml(dataset_info_file)
        return cls(**dataset_info.to_dict(), root_dir=root_dir)

    @classmethod
    def from_coco(
        cls,
        name: str,
        task: str,
        coco_file: str,
        coco_root_dir: str,
        root_dir: str = None,
    ) -> "Dataset":
        """Import Dataset from coco format.

        Args:
            name (str): Dataset name.
            task (str): Dataset task.
            coco_file (str): Coco json file path.
            coco_root_dir (str): Coco image root directory.
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Raises:
            FileExistsError: if new dataset name already exist.

        Returns:
            Dataset: Dataset Class
        """
        ds = Dataset.new(name, task, root_dir)
        ds.initialize()

        # parse coco annotation file
        coco = io.load_json(coco_file)
        for image_dict in coco["images"]:
            image_id = image_dict.pop("id")
            ds.add_images(
                [Image.from_dict({**image_dict, "image_id": image_id})]
            )
        for annotation_dict in coco["annotations"]:
            annotation_id = annotation_dict.pop("id")
            ds.add_annotations(
                [
                    Annotation.from_dict(
                        {**annotation_dict, "annotation_id": annotation_id},
                        task,
                    )
                ]
            )
        for category_dict in coco["categories"]:
            category_id = category_dict.pop("id")
            ds.add_categories(
                [
                    Category.from_dict(
                        {**category_dict, "category_id": category_id}, task
                    )
                ]
            )

        # copy raw images
        io.copy_files_to_directory(coco_root_dir, ds.raw_image_dir)

        return ds

    @classmethod
    def from_huggingface(
        cls,
        name: str,
        task: str,
        dataset_dir: str,
        root_dir=None,
    ) -> "Dataset":
        """Import Dataset from huggingface datasets.

        Args:
            name (str): Dataset name.
            dataset_dir (str): Hugging Face dataset directory.
            task (str): Task name. Task must be one of ["classification", "object_detection"]
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Raises:
            FileExistsError: if dataset name already exists
            ValueError: if dataset is not Dataset or DatasetDict

        Returns:
            Dataset: Dataset Class
        """
        ds = cls(name, root_dir)
        if ds.initialized():
            raise FileExistsError(
                f"{ds.dataset_dir} already exists. try another name."
            )
        ds.initialize()

        dataset = load_from_disk(dataset_dir)

        if isinstance(dataset, DatasetDict):
            is_splited = True
        elif isinstance(dataset, HFDataset):
            is_splited = False
        else:
            raise ValueError("dataset should be Dataset or DatasetDict")

        def _import(dataset: HFDataset, task: str):
            if task == "object_detection":
                for data in dataset:
                    data["image"].save(
                        f"{ds.raw_image_dir}/{data['image_id']}.jpg"
                    )
                    image = Image(
                        image_id=data["image_id"],
                        file_name=f"{data['image_id']}.jpg",
                        width=data["width"],
                        height=data["height"],
                    )
                    ds.add_images([image])

                    annotation_ids = data["objects"]["id"]
                    areas = data["objects"]["area"]
                    category_ids = data["objects"]["category"]
                    bboxes = data["objects"]["bbox"]

                    for annotation_id, area, category_id, bbox in zip(
                        annotation_ids, areas, category_ids, bboxes
                    ):
                        annotation = Annotation(
                            annotation_id=annotation_id,
                            image_id=image.image_id,
                            category_id=category_id + 1,
                            area=area,
                            bbox=bbox,
                        )
                        ds.add_annotations([annotation])

                categories = (
                    dataset.features["objects"].feature["category"].names
                )
                for category_id, category_name in enumerate(categories):
                    category = Category(
                        category_id=category_id + 1,
                        supercategory="object",
                        name=category_name,
                    )
                    ds.add_categories([category])

            elif task == "classification":
                for data in dataset:
                    data["image"].save(
                        f"{ds.raw_image_dir}/{data['image_id']}.jpg"
                    )
                    image = Image(
                        image_id=data["image_id"],
                        file_name=f"{data['image_id']}.jpg",
                        width=data["width"],
                        height=data["height"],
                    )
                    ds.add_images([image])

                    annotation = Annotation(
                        annotation_id=data["image_id"],
                        image_id=image.image_id,
                        category_id=data["label"] + 1,
                    )
                    ds.add_annotations([annotation])

                categories = dataset.features["label"].names
                for category_id, category_name in enumerate(categories):
                    category = Category(
                        category_id=category_id + 1,
                        supercategory="object",
                        name=category_name,
                    )
                    ds.add_categories([category])
            else:
                raise ValueError(
                    "task should be one of ['classification', 'object_detection']"
                )

        if is_splited:
            for set_type, set in dataset.items():
                image_ids = set["image_id"]
                io.save_json(image_ids, ds.set_dir / f"{set_type}.json", True)
                _import(set, task)
        else:
            _import(dataset, task)

        return ds

    def initialize(self):
        """Initialize Dataset.
        It creates necessary directories under {dataset_root_dir}/{dataset_name}.
        """
        io.make_directory(self.raw_image_dir)
        io.make_directory(self.image_dir)
        io.make_directory(self.annotation_dir)
        io.make_directory(self.category_dir)

        # create dataset_info.yaml
        io.save_yaml(
            DatasetInfo(
                name=self.name,
                task=self.task,
            ).to_dict(),
            self.dataset_info_file,
        )

    def initialized(self) -> bool:
        """Check if Dataset has been initialized or not.

        Returns:
            bool:
                initialized -> True
                not initialized -> False
        """
        return self.dataset_info_file.exists()

    # get
    def get_images(
        self, image_ids: list[int] = None, labeled: bool = True
    ) -> list[Image]:
        """Get "Image"s.

        Args:
            image_ids (list[int], optional): id list. None for all "Image"s. Defaults to None.
            labeled (bool, optional): get labeled images. False for unlabeled images. Defaults to True.

        Returns:
            list[Image]: "Image" list
        """
        image_files = (
            list(map(lambda x: self.image_dir / (str(x) + ".json"), image_ids))
            if image_ids
            else list(self.image_dir.glob("*.json"))
        )
        labeled_images = []
        unlabeled_images = []
        for image_file in image_files:
            if self.get_annotations(image_file.stem):
                labeled_images.append(Image.from_json(image_file))
            else:
                unlabeled_images.append(Image.from_json(image_file))

        if labeled:
            logger.info(f"Found {len(labeled_images)} labeled images")
            return labeled_images
        else:
            logger.info(f"Found {len(unlabeled_images)} unlabeled images")
            return unlabeled_images

    def get_categories(self, category_ids: list[int] = None) -> list[Category]:
        """Get "Category"s.

        Args:
            category_ids (list[int], optional): id list. None for all "Category"s. Defaults to None.

        Returns:
            list[Category]: "Category" list
        """
        return [
            Category.from_json(f, self.task)
            for f in (
                [
                    self.category_dir / f"{category_id}.json"
                    for category_id in category_ids
                ]
                if category_ids
                else self.category_dir.glob("*.json")
            )
        ]

    def get_annotations(self, image_id: int = None) -> list[Annotation]:
        """Get "Annotation"s.

        Args:
            image_id (int, optional): image id. None for all "Annotation"s. Defaults to None.

        Returns:
            list[Annotation]: "Annotation" list
        """
        if image_id:
            return [
                Annotation.from_json(f, self.task)
                for f in self.annotation_dir.glob(f"{image_id}/*.json")
            ]
        else:
            return [
                Annotation.from_json(f, self.task)
                for f in self.annotation_dir.glob("*/*.json")
            ]

    def get_predictions(self, image_id: int = None) -> list[Annotation]:
        """Get "Prediction"s.

        Args:
            image_id (int, optional): image id. None for all "Prediction"s. Defaults to None.

        Returns:
            list[Annotation]: "Prediction" list
        """
        if image_id:
            return [
                Annotation.from_json(f, self.task)
                for f in self.prediction_dir.glob(f"{image_id}/*.json")
            ]
        else:
            return [
                Annotation.from_json(f, self.task)
                for f in self.prediction_dir.glob("*/*.json")
            ]

    # add
    def add_images(self, images: list[Image]):
        """Add "Image"s to dataset.

        Args:
            images (list[Image]): list of "Image"s
        """
        for item in images:
            item_id = item.image_id
            item_path = self.image_dir / f"{item_id}.json"
            io.save_json(item.to_dict(), item_path)

    def add_categories(self, categories: list[Category]):
        """Add "Category"s to dataset.

        Args:
            categories (list[Category]): list of "Category"s
        """
        for item in categories:
            item_id = item.category_id
            item_path = self.category_dir / f"{item_id}.json"
            io.save_json(item.to_dict(), item_path)

    def add_annotations(self, annotations: list[Annotation]):
        """Add "Annotation"s to dataset.

        Args:
            annotations (list[Annotation]): list of "Annotation"s
        """
        for item in annotations:
            item_path = (
                self.annotation_dir
                / f"{item.image_id}"
                / f"{item.annotation_id}.json"
            )
            io.save_json(item.to_dict(), item_path, create_directory=True)

    def add_predictions(self, predictions: list[Annotation]):
        """Add "Annotation"s to dataset.

        Args:
            annotations (list[Annotation]): list of "Annotation"s
        """
        for item in predictions:
            item_path = (
                self.prediction_dir
                / f"{item.image_id}"
                / f"{item.annotation_id}.json"
            )
            io.save_json(item.to_dict(), item_path, create_directory=True)

    # functions
    def split(
        self,
        train_ratio: float,
        val_ratio: float = 0.0,
        test_ratio: float = 0.0,
        seed: int = 0,
    ):
        """Split Dataset to train, validation, test, (unlabeled) sets.

        Args:
            train_ratio (float): train num ratio (0 ~ 1).
            val_ratio (float, optional): val num ratio (0 ~ 1).
            test_ratio (float, optional): test num ratio (0 ~ 1).
            seed (int, optional): random seed. Defaults to 0.
        """

        if val_ratio == 0.0:
            val_ratio = 1 - train_ratio

        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio

        images: list[Image] = self.get_images(labeled=True)

        num_images = len(images)

        random.seed(seed)
        random.shuffle(images)

        train_num = round(num_images * train_ratio)
        val_num = round(num_images * val_ratio)
        test_num = round(num_images * test_ratio)
        logger.info(f"train: {train_num}, val: {val_num}, test: {test_num}")

        train_images = images[:train_num]
        val_images = images[train_num : train_num + val_num]
        test_images = (
            images[train_num + val_num :] if test_num > 0 else val_images
        )

        io.save_json(
            list(map(lambda x: x.image_id, train_images)),
            self.train_set_file,
            create_directory=True,
        )
        io.save_json(
            list(map(lambda x: x.image_id, val_images)),
            self.val_set_file,
            create_directory=True,
        )
        io.save_json(
            list(map(lambda x: x.image_id, test_images)),
            self.test_set_file,
            create_directory=True,
        )

        unlabeled_images: list[Image] = self.get_images(labeled=False)

        io.save_json(
            list(map(lambda x: x.image_id, unlabeled_images)),
            self.unlabeled_set_file,
            create_directory=True,
        )

    # export
    def get_split_ids(self) -> list[list[int]]:
        """Get split ids

        Returns:
            list[list[int]]: split ids
        """
        if not self.train_set_file.exists():
            raise FileNotFoundError(
                "There is no set files. Please run ds.split() first"
            )

        train_ids: list[int] = io.load_json(self.train_set_file)
        val_ids: list[int] = io.load_json(self.val_set_file)
        test_ids: list[int] = io.load_json(self.test_set_file)
        unlabeled_ids: list[int] = io.load_json(self.unlabeled_set_file)

        return [train_ids, val_ids, test_ids, unlabeled_ids]

    def export(self, data_type: Union[str, DataType]) -> str:
        """Export Dataset to Specific data formats

        Args:
            data_type (Union[str, DataType]): export data type. one of ["YOLO", "COCO"].

        Returns:
            str: exported dataset directory
        """
        export_dir: Path = self.export_dir / str(data_type)
        if export_dir.exists():
            io.remove_directory(export_dir)
            warnings.warn(
                f"{export_dir} already exists. Removing exist export and override."
            )

        try:
            if data_type == DataType.YOLO:
                export_dir = export_yolo(self, export_dir)
            elif data_type == DataType.COCO:
                export_dir = export_coco(self, export_dir)
            elif data_type == DataType.HUGGINGFACE:
                export_dir = export_huggingface(self, export_dir)
            else:
                raise ValueError(f"Invalid data_type: {data_type}")

            return export_dir
        except Exception as e:
            io.remove_directory(export_dir)
            raise e

    def delete(self):
        """Delete Dataset"""
        io.remove_directory(self.dataset_dir)
        del self
