import copy
import logging
import os
import random
import shutil
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from functools import cached_property
from pathlib import Path
from tempfile import mkdtemp
from typing import Union

import PIL.Image
import tqdm
from waffle_utils.file import io, network
from waffle_utils.image.io import load_image, save_image
from waffle_utils.log import datetime_now
from waffle_utils.utils import type_validator

from waffle_hub import EXPORT_MAP, DataType, SplitMethod, TaskType
from waffle_hub.dataset.adapter import (
    export_autocare_dlt,
    export_coco,
    export_transformers,
    export_yolo,
    import_autocare_dlt,
    import_coco,
    import_label_studio,
    import_transformers,
    import_yolo,
)
from waffle_hub.schema import Annotation, Category, DatasetInfo, Image
from waffle_hub.utils.draw import draw_results

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
    DRAW_DIR = Path("draws")

    TRAIN_SET_FILE_NAME = Path("train.json")
    VAL_SET_FILE_NAME = Path("val.json")
    TEST_SET_FILE_NAME = Path("test.json")
    UNLABELED_SET_FILE_NAME = Path("unlabeled.json")

    MINIMUM_TRAINABLE_IMAGE_NUM_PER_CATEGORY = 3

    def __init__(
        self,
        name: str,
        task: Union[str, TaskType],
        categories: list[Union[str, int, float, dict, Category]] = None,
        created: str = None,
        root_dir: str = None,
    ):
        self.name = name
        self.task = task
        self.created = created

        self.root_dir = root_dir

        if not self.initialized():
            self._initialize()
            self._set_categories(categories)
            self.save_dataset_info()
        else:  # for backward compatibility
            self.save_dataset_info()

    def __repr__(self):
        return self.get_dataset_info().__repr__()

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
        v = str(v).upper()
        if v not in TaskType:
            raise ValueError(f"Invalid task type: {v}" f"Available task types: {list(TaskType)}")
        self.__task = v

    @property
    def categories(self) -> list[Category]:
        return self.get_categories()

    def _set_categories(self, v):
        if v is None or len(v) == 0:
            v = []
        elif isinstance(v[0], dict):
            v = [
                getattr(Category, self.task.lower())(
                    **{
                        **category,
                        "category_id": category.get("category_id", i),
                    }
                )
                for i, category in enumerate(v, start=1)
            ]
        elif isinstance(v[0], (str, int, float)):
            v = [
                getattr(Category, self.task.lower())(
                    category_id=i,
                    supercategory="object",
                    name=str(category),
                )
                for i, category in enumerate(v, start=1)
            ]
        elif isinstance(v[0], Category):
            pass

        self.add_categories(v)

    def extract_by_image_ids(
        self, new_name: str, image_ids: list[int], root_dir: str = None
    ) -> "Dataset":
        """
        Extract a new dataset by image ids

        Args:
            new_name (str): Name of the new dataset
            image_ids (list[int]): Image ids to extract
            root_dir (str, optional): Root directory of the new dataset. Defaults to None.

        Returns:
            Dataset: Extracted dataset

        """
        ds = Dataset.new(
            name=new_name,
            task=self.task,
            root_dir=root_dir,
        )

        try:
            ds.add_categories(self.get_categories())
            for image in self.get_images(image_ids):
                annotations = self.get_annotations(image.image_id)
                io.copy_file(
                    self.raw_image_dir / image.file_name, ds.raw_image_dir / image.file_name
                )
                ds.add_images([image])
                ds.add_annotations(annotations)

        except Exception as e:
            ds.delete()
            raise e

        return ds

    def extract_by_categories(
        self, new_name: str, category_ids: list[int], root_dir: str = None
    ) -> "Dataset":
        """
        Extract a new dataset by categories

        Args:
            new_name (str): Name of the new dataset
            category_ids (list[int]): Category IDs to extract
            root_dir (str, optional): Root directory of the new dataset. Defaults to None.

        Returns (Dataset): New dataset
        """
        ds = Dataset.new(
            name=new_name,
            task=self.task,
            root_dir=root_dir,
        )
        try:
            category_old2new = {}
            for new_category_id, category_id in enumerate(category_ids, start=1):
                category_old2new[category_id] = new_category_id
                categories = self.get_categories([category_id])
                for category in categories:
                    category.category_id = new_category_id
                ds.add_categories(categories)

            for image in self.get_images():
                annotations = list(
                    filter(
                        lambda ann: ann.category_id in category_ids,
                        self.get_annotations(image.image_id),
                    )
                )
                for annotation in annotations:
                    annotation.category_id = category_old2new[annotation.category_id]

                if annotations:
                    io.copy_file(
                        self.raw_image_dir / image.file_name, ds.raw_image_dir / image.file_name
                    )
                    ds.add_images([image])
                    ds.add_annotations(annotations)

        except Exception as e:
            ds.delete()
            raise e

        return ds

    @property
    def created(self):
        return self.__created

    @created.setter
    def created(self, v):
        self.__created = v or datetime_now()

    @property
    def root_dir(self) -> Path:
        return self.__root_dir

    @root_dir.setter
    @type_validator(Path, strict=False)
    def root_dir(self, v):
        self.__root_dir = Dataset.parse_root_dir(v)
        logger.info(f"Dataset root directory: {self.__root_dir}")

    @classmethod
    def parse_root_dir(cls, v):
        if v:
            return Path(v)
        elif os.getenv("WAFFLE_DATASET_ROOT_DIR", None):
            return Path(os.getenv("WAFFLE_DATASET_ROOT_DIR"))
        else:
            return Dataset.DEFAULT_DATASET_ROOT_DIR

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
    def draw_dir(self) -> Path:
        return self.dataset_dir / Dataset.DRAW_DIR

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

    # dataset indexes
    @property
    def image_dict(self) -> dict[int, Image]:
        if not hasattr(self, "_image_dict"):
            self.create_index()
        return self._image_dict

    @property
    def unlabeled_image_dict(self) -> dict[int, Image]:
        if not hasattr(self, "_unlabeled_image_dict"):
            self.create_index()
        return self._unlabeled_image_dict

    @property
    def annotation_dict(self) -> dict[int, Annotation]:
        if not hasattr(self, "_annotation_dict"):
            self.create_index()
        return self._annotation_dict

    @property
    def prediction_dict(self) -> dict[int, Annotation]:
        if not hasattr(self, "_prediction_dict"):
            self.create_index()
        return self._prediction_dict

    @property
    def category_dict(self) -> dict[int, Category]:
        if not hasattr(self, "_category_dict"):
            self.create_index()
        return self._category_dict

    @property
    def image_to_annotations(self) -> dict[int, list[Annotation]]:
        if not hasattr(self, "_image_to_annotations"):
            self.create_index()
        return self._image_to_annotations

    @property
    def image_to_predictions(self) -> dict[int, list[Annotation]]:
        if not hasattr(self, "_image_to_predictions"):
            self.create_index()
        return self._image_to_predictions

    @property
    def annotation_to_image(self) -> dict[int, Image]:
        if not hasattr(self, "_annotation_to_image"):
            self.create_index()
        return self._annotation_to_image

    @property
    def prediction_to_image(self) -> dict[int, Image]:
        if not hasattr(self, "_prediction_to_image"):
            self.create_index()
        return self._prediction_to_image

    @property
    def category_to_images(self) -> dict[int, list[Image]]:
        if not hasattr(self, "_category_to_images"):
            self.create_index()
        return self._category_to_images

    @property
    def category_to_unique_images(self) -> dict[int, list[Image]]:
        if not hasattr(self, "_category_to_unique_images"):
            self.create_index()
        return self._category_to_unique_images

    @property
    def category_name_to_category(self) -> dict[str, Category]:
        if not hasattr(self, "_category_name_to_category"):
            self.create_index()
        return self._category_name_to_category

    @property
    def category_to_annotations(self) -> dict[int, list[Annotation]]:
        if not hasattr(self, "_category_to_annotations"):
            self.create_index()
        return self._category_to_annotations

    @property
    def category_to_predictions(self) -> dict[int, list[Annotation]]:
        if not hasattr(self, "_category_to_predictions"):
            self.create_index()
        return self._category_to_predictions

    def get_category_names(self) -> list[str]:
        return [category.name for category in self.categories]

    # factories
    @classmethod
    def new(
        cls,
        name: str,
        task: str,
        categories: list[Union[str, int, float, dict, Category]] = None,
        root_dir: str = None,
    ) -> "Dataset":
        """
        Create New Dataset.
        This method creates a new dataset directory and initialize dataset info file.
        If you have other types of data, you can use from_* methods to create a dataset.

        Args:
            name (str): Dataset name
            task (str): Dataset task
            categories (list[Union[str, int, float, dict, Category]]): Dataset categories
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Raises:
            FileExistsError: if dataset name already exists

        Examples:
            >>> ds = Dataset.new("my_dataset", "CLASSIFICATION")
            >>> ds.name
            'my_dataset'  # dataset name
            >>> ds.task  # dataset task
            'CLASSIFICATION'

        Returns:
            Dataset: Dataset Class
        """
        root_dir = Dataset.parse_root_dir(root_dir)

        if name in Dataset.get_dataset_list(root_dir):
            raise FileExistsError(f"Dataset {name} already exists.")

        try:
            return cls(name=name, task=task, categories=categories, root_dir=root_dir)
        except Exception as e:
            if (root_dir / name).exists():
                io.remove_directory(root_dir / name)
            raise e

    @classmethod
    def clone(
        cls,
        src_name: str,
        name: str,
        src_root_dir: str = None,
        root_dir: str = None,
    ) -> "Dataset":
        """
        Clone Existing Dataset.
        This method clones an existing dataset.

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

        Examples:
            >>> ds = Dataset.clone("my_dataset", "my_dataset_clone")
            >>> ds.name
            'my_dataset_clone'  # cloned dataset name
            >>> ds.task
            'CLASSIFICATION'   # original dataset task

        Returns:
            Dataset: Dataset Class
        """
        root_dir = Dataset.parse_root_dir(root_dir)

        try:
            src_ds = Dataset.load(src_name, src_root_dir)

            ds = Dataset.new(name=name, task=src_ds.task, root_dir=root_dir)
            io.copy_files_to_directory(src_ds.dataset_dir, ds.dataset_dir, create_directory=True)
            ds.save_dataset_info()

            ds.create_index()
            return ds
        except Exception as e:
            if (root_dir / name).exists():
                io.remove_directory(root_dir / name)
            raise e

    @classmethod
    def dummy(
        cls,
        name: str,
        task: str,
        image_num: int = 100,
        category_num: int = 10,
        unlabeled_image_num: int = 0,
        root_dir: str = None,
    ) -> "Dataset":
        """
        Create Dummy Dataset (for debugging).

        Args:
            name (str): Dataset name
            task (str): Dataset task
            image_num (int, optional): Number of images. Defaults to 100.
            category_num (int, optional): Number of categories. Defaults to 10.
            unlabeld_image_num (int, optional): Number of unlabeled images. Defaults to 0.
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Raises:
            FileExistsError: if dataset name already exists

        Examples:
            >>> ds = Dataset.dummy("my_dataset", "CLASSIFICATION", image_num=100, category_num=10)
            >>> len(ds.get_images())
            100
            >>> len(ds.get_categories())
            10
        """
        ds = Dataset.new(name=name, task=task, root_dir=root_dir)

        try:
            for category_id in range(1, category_num + 1):

                if task == TaskType.CLASSIFICATION:
                    category = Category.classification(
                        category_id=category_id,
                        name=f"category_{category_id}",
                        supercategory="object",
                    )
                elif task == TaskType.OBJECT_DETECTION:
                    category = Category.object_detection(
                        category_id=category_id,
                        name=f"category_{category_id}",
                        supercategory="object",
                    )
                elif task == TaskType.INSTANCE_SEGMENTATION:
                    category = Category.instance_segmentation(
                        category_id=category_id,
                        name=f"category_{category_id}",
                        supercategory="object",
                    )
                elif task == TaskType.TEXT_RECOGNITION:
                    category = Category.text_recognition(
                        category_id=category_id,
                        name=chr(64 + category_id),
                        supercategory="object",
                    )

                ds.add_categories([category])

            annotation_id = 1
            for image_id in range(1, image_num + 1):
                file_name = f"image_{image_id}.jpg"
                ds.add_images([Image(image_id=image_id, file_name=file_name, width=100, height=100)])
                PIL.Image.new("RGB", (100, 100)).save(ds.raw_image_dir / file_name)

                if task == TaskType.CLASSIFICATION:
                    annotations = [
                        Annotation.classification(
                            annotation_id=annotation_id,
                            image_id=image_id,
                            category_id=random.randint(1, category_num),
                        )
                    ]
                elif task == TaskType.OBJECT_DETECTION:
                    annotations = [
                        Annotation.object_detection(
                            annotation_id=annotation_id + i,
                            image_id=image_id,
                            category_id=random.randint(1, category_num),
                            bbox=[
                                random.randint(0, 100),
                                random.randint(0, 100),
                                random.randint(0, 100),
                                random.randint(0, 100),
                            ],
                        )
                        for i in range(random.randint(1, 5))
                    ]
                elif task == TaskType.INSTANCE_SEGMENTATION:
                    annotations = [
                        Annotation.instance_segmentation(
                            annotation_id=annotation_id + i,
                            image_id=image_id,
                            category_id=random.randint(1, category_num),
                            bbox=[
                                random.randint(0, 100),
                                random.randint(0, 100),
                                random.randint(0, 100),
                                random.randint(0, 100),
                            ],
                            segmentation=[[random.randint(0, 100) for _ in range(10)]],
                        )
                        for i in range(random.randint(1, 5))
                    ]
                elif task == TaskType.TEXT_RECOGNITION:
                    annotations = [
                        Annotation.text_recognition(
                            annotation_id=annotation_id,
                            image_id=image_id,
                            caption=chr(64 + random.randint(1, category_num)),
                        )
                    ]
                ds.add_annotations(annotations)
                annotation_id += len(annotations)

            if unlabeled_image_num > 0:
                for image_id in range(image_num + 1, image_num + unlabeled_image_num + 1):
                    file_name = f"image_{image_id}.jpg"
                    ds.add_images(
                        [Image(image_id=image_id, file_name=file_name, width=100, height=100)]
                    )
                    PIL.Image.new("RGB", (100, 100)).save(ds.raw_image_dir / file_name)

        except Exception as e:
            ds.delete()
            raise e

        ds.create_index()
        return ds

    @classmethod
    def load(cls, name: str, root_dir: str = None) -> "Dataset":
        """
        Load Dataset.
        This method loads an existing dataset.

        Args:
            name (str): Dataset name that Waffle Created
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Raises:
            FileNotFoundError: if source dataset does not exist.

        Examples:
            >>> ds = Dataset.load("my_dataset")
            >>> ds.name
            'my_dataset'  # dataset name

        Returns:
            Dataset: Dataset Class
        """
        root_dir = Dataset.parse_root_dir(root_dir)
        dataset_info_file = root_dir / name / Dataset.DATASET_INFO_FILE_NAME
        if not dataset_info_file.exists():
            raise FileNotFoundError(f"{dataset_info_file} has not been created.")
        dataset_info = DatasetInfo.load(dataset_info_file)

        ds = cls(**dataset_info.to_dict(), root_dir=root_dir)
        ds.create_index()
        return ds

    @classmethod
    def merge(
        cls,
        name: str,
        root_dir: str,
        src_names: list[str],
        src_root_dirs: Union[str, list[str]],
        task: str,
    ) -> "Dataset":
        """
        Merge Datasets.
        This method merges multiple datasets into one dataset.

        Args:
            name (str): New Dataset name
            root_dir (str): New Dataset root directory
            src_names (list[str]): Source Dataset names
            src_root_dirs (Union[str, list[str]]): Source Dataset root directories
            task (str): Dataset task

        Returns:
            Dataset: Dataset Class

        """
        if isinstance(src_root_dirs, str):
            src_root_dirs = [src_root_dirs] * len(src_names)
        if len(src_names) != len(src_root_dirs):
            raise ValueError("Length of src_names and src_root_dirs should be same.")
        if isinstance(task, str):
            task = task.upper()
        if task not in [k for k in TaskType]:
            raise ValueError(f"task should be one of {[k for k in TaskType]}")

        merged_ds = Dataset.new(
            name=name,
            root_dir=root_dir,
            task=task,
        )

        categoryname2id = {}
        filename2id = {}
        new_annotation_id = 1

        try:
            for src_name, src_root_dir in zip(src_names, src_root_dirs):
                src_ds = Dataset.load(src_name, src_root_dir)
                src_categories = src_ds.get_categories()

                if src_ds.task != task:
                    raise ValueError(f"Task of {src_ds.name} is {src_ds.task}. It should be {task}.")

                # merge - raw images
                io.copy_files_to_directory(
                    src_ds.raw_image_dir, merged_ds.raw_image_dir, create_directory=True
                )

                # merge - categories
                for category in src_ds.get_categories():
                    if category.name not in categoryname2id:
                        new_category_id = len(categoryname2id) + 1
                        categoryname2id[category.name] = new_category_id

                        new_category = copy.deepcopy(category)
                        new_category.category_id = new_category_id
                        merged_ds.add_categories([new_category])

                for image_id, annotations in src_ds.image_to_annotations.items():
                    image = src_ds.get_images([image_id])[0]

                    # merge - images
                    is_new_image = False
                    if image.file_name not in filename2id:
                        is_new_image = True

                        new_image_id = len(filename2id) + 1
                        filename2id[image.file_name] = new_image_id

                        new_image = copy.deepcopy(image)
                        new_image.image_id = new_image_id
                        merged_ds.add_images([new_image])

                    new_image_id = filename2id[image.file_name]

                    # merge - annotations
                    for annotation in annotations:
                        new_annotation = copy.deepcopy(annotation)
                        new_annotation.category_id = categoryname2id[
                            src_categories[annotation.category_id - 1].name
                        ]

                        # check if new annotation
                        is_new_annotation = True
                        if not is_new_image:
                            for merged_ann in merged_ds.get_annotations(new_image_id):
                                if new_annotation == merged_ann:
                                    is_new_annotation = False
                                    break

                        # merge
                        if is_new_annotation:
                            new_annotation.image_id = filename2id[image.file_name]
                            new_annotation.annotation_id = new_annotation_id
                            new_annotation_id += 1
                            merged_ds.add_annotations([new_annotation])

        except Exception as e:
            if merged_ds.dataset_dir.exists():
                io.remove_directory(merged_ds.dataset_dir)
            raise e

        ds = Dataset.load(name, root_dir)
        ds.create_index()
        return ds

    @classmethod
    def from_coco(
        cls,
        name: str,
        task: str,
        coco_file: Union[str, list[str]],
        coco_root_dir: Union[str, list[str]],
        root_dir: str = None,
    ) -> "Dataset":
        """
        Import Dataset from coco format.
        This method imports coco format dataset.

        Args:
            name (str): Dataset name.
            task (str): Dataset task.
            coco_file (Union[str, list[str]]): Coco json file path. If given list, it will be regarded as [train, val, test] json file.
            coco_root_dir (Union[str, list[str]]): Coco image root directory. If given list, it will be regarded as [train, val, test] coco root file.
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Raises:
            FileExistsError: if new dataset name already exist.

        Examples:
            # Import one coco json file.
            >>> ds = Dataset.from_coco("my_dataset", "object_detection", "path/to/coco.json", "path/to/coco_root")
            >>> ds.get_images()
            {<Image: 1>, <Image: 2>, <Image: 3>, <Image: 4>, <Image: 5>}
            >>> ds.get_annotations()
            {<Annotation: 1>, <Annotation: 2>, <Annotation: 3>, <Annotation: 4>, <Annotation: 5>}
            >>> ds.get_categories()
            {<Category: 1>, <Category: 2>, <Category: 3>, <Category: 4>, <Category: 5>}
            >>> ds.get_category_names()
            ['person', 'bicycle', 'car', 'motorcycle', 'airplane']

            # Import multiple coco json files.
            # You can give coco_file as list.
            # Given coco files are regarded as [train, [val, [test]]] json files.
            >>> ds = Dataset.from_coco("my_dataset", "object_detection", ["coco_train.json", "coco_val.json"], ["coco_train_root", "coco_val_root"])

        Returns:
            Dataset: Dataset Class
        """
        ds = Dataset.new(name=name, task=task, root_dir=root_dir)

        try:
            if isinstance(coco_file, list) and isinstance(coco_root_dir, list):
                if len(coco_file) != len(coco_root_dir):
                    raise ValueError("coco_file and coco_root_dir should have same length.")
            if not isinstance(coco_file, list) and isinstance(coco_root_dir, list):
                raise ValueError(
                    "ambiguous input. The number of coco_file should be same or greater than coco_root_dir."
                )
            if not isinstance(coco_file, list):
                coco_file = [coco_file]
            if not isinstance(coco_root_dir, list):
                coco_root_dir = [coco_root_dir] * len(coco_file)

            coco_files = coco_file
            coco_root_dirs = coco_root_dir

            import_coco(ds, coco_files, coco_root_dirs)

            if len(coco_files) == 2:
                logger.info("copying val set to test set")
                io.copy_file(ds.val_set_file, ds.test_set_file, create_directory=True)

            # TODO: add unlabeled set
            io.save_json([], ds.unlabeled_set_file, create_directory=True)

        except Exception as e:
            ds.delete()
            raise e

        ds.create_index()
        return ds

    @classmethod
    def from_autocare_dlt(
        cls,
        name: str,
        task: str,
        coco_file: Union[str, list[str]],
        coco_root_dir: Union[str, list[str]],
        root_dir: str = None,
    ) -> "Dataset":
        """
        Import dataset from autocare dlt format.
        This method is used for importing dataset from autocare dlt format.

        Args:
            name (str): name of dataset.
            task (str): task of dataset.
            coco_file (Union[str, list[str]]): coco annotation file path.
            coco_root_dir (Union[str, list[str]]): root directory of coco dataset.
            root_dir (str, optional): root directory of dataset. Defaults to None.

        Raises:
            FileExistsError: if new dataset name already exist.

        Examples:
            # Import one coco json file.
            >>> ds = Dataset.from_coco("my_dataset", "object_detection", "path/to/coco.json", "path/to/coco_root")
            >>> ds.get_images()
            {<Image: 1>, <Image: 2>, <Image: 3>, <Image: 4>, <Image: 5>}
            >>> ds.get_annotations()
            {<Annotation: 1>, <Annotation: 2>, <Annotation: 3>, <Annotation: 4>, <Annotation: 5>}
            >>> ds.get_categories()
            {<Category: 1>, <Category: 2>, <Category: 3>, <Category: 4>, <Category: 5>}
            >>> ds.get_category_names()
            ['person', 'bicycle', 'car', 'motorcycle', 'airplane']

        Returns:
            Dataset: Dataset Class.
        """
        ds = Dataset.new(name=name, task=task, root_dir=root_dir)

        try:
            if isinstance(coco_file, list) and isinstance(coco_root_dir, list):
                if len(coco_file) != len(coco_root_dir):
                    raise ValueError("coco_file and coco_root_dir should have same length.")
            if not isinstance(coco_file, list) and isinstance(coco_root_dir, list):
                raise ValueError(
                    "ambiguous input. The number of coco_file should be same or greater than coco_root_dir."
                )
            if not isinstance(coco_file, list):
                coco_file = [coco_file]
            if not isinstance(coco_root_dir, list):
                coco_root_dir = [coco_root_dir] * len(coco_file)

            coco_files = coco_file
            coco_root_dirs = coco_root_dir

            import_autocare_dlt(ds, coco_files, coco_root_dirs)

            if len(coco_files) == 2:
                logging.info("copying val set to test set")
                io.copy_file(ds.val_set_file, ds.test_set_file, create_directory=True)

            # TODO: add unlabeled set
            io.save_json([], ds.unlabeled_set_file, create_directory=True)

        except Exception as e:
            ds.delete()
            raise e

        ds.create_index()
        return ds

    @classmethod
    def from_yolo(
        cls,
        name: str,
        task: str,
        yolo_root_dir: str,
        yaml_path: str = None,
        root_dir: str = None,
    ) -> "Dataset":
        """
        Import Dataset from yolo format.
        This method imports dataset from yolo(ultralytics) yaml file.

        Args:
            name (str): Dataset name.
            task (str): Dataset task.
            yolo_root_dir (str): Yolo dataset root directory.
            yaml_path (str): Yolo yaml file path. when task is classification, yaml_path is not required.
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Example:
            >>> ds = Dataset.from_yolo("yolo", "classification", "path/to/yolo.yaml")

        Returns:
            Dataset: Imported dataset.
        """

        ds = Dataset.new(name=name, task=task, root_dir=root_dir)

        try:
            import_yolo(ds, yolo_root_dir, yaml_path)

        except Exception as e:
            ds.delete()
            raise e

        ds.create_index()
        return ds

    @classmethod
    def from_transformers(
        cls,
        name: str,
        task: str,
        dataset_dir: str,
        root_dir=None,
    ) -> "Dataset":
        """
        Import Dataset from transformers datasets.
        This method imports transformers dataset from directory.

        Args:
            name (str): Dataset name.
            dataset_dir (str): Transformers dataset directory.
            task (str): Task name.
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Raises:
            FileExistsError: if dataset name already exists
            ValueError: if dataset is not Dataset or DatasetDict

        Examples:
            >>> ds = Dataset.from_transformers("transformers", "object_detection", "path/to/transformers/dataset")

        Returns:
            Dataset: Dataset Class
        """
        ds = Dataset.new(name=name, task=task, root_dir=root_dir)

        try:
            import_transformers(ds, dataset_dir)

            # TODO: add unlabeled set
            io.save_json([], ds.unlabeled_set_file, create_directory=True)

        except Exception as e:
            ds.delete()
            raise e

        ds.create_index()
        return ds

    @classmethod
    def from_label_studio(
        cls,
        name: str,
        task: str,
        json_file: str,
        image_dir: str = None,
        root_dir: str = None,
    ) -> "Dataset":
        """
        Import Dataset from label_studio format.
        This method imports dataset from label_studio exported json file (the first one).

        Args:
            name (str): Dataset name.
            task (str): Dataset task.
            json_file (str): Label studio json file path.
            image_dir (str): Label studio image directory.
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Example:
            >>> ds = Dataset.from_label_studio(
                "label_studio",
                "classification",
                "path/to/label_studio/json/export/file.json",
                "path/to/image_dir"
            )

        Returns:
            Dataset: Imported dataset.
        """

        ds = Dataset.new(name=name, task=task, root_dir=root_dir)

        try:
            import_label_studio(
                self=ds,
                json_file=json_file,
                task=task,
                image_dir=image_dir,
            )

        except Exception as e:
            ds.delete()
            raise e

        ds.create_index()
        return ds

    @classmethod
    def sample(cls, name: str, task: str, root_dir: str = None) -> "Dataset":
        """
        Import sample Dataset.

        Args:
            name (str): Dataset name.
            task (str): Task name.
            root_dir (str, optional): Dataset root directory. Defaults to None.

        Returns:
            Dataset: Dataset Class
        """

        temp_dir = Path(mkdtemp())
        try:
            if task in [
                TaskType.CLASSIFICATION,
                TaskType.OBJECT_DETECTION,
                TaskType.INSTANCE_SEGMENTATION,
            ]:
                url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist.zip"
            elif task == TaskType.TEXT_RECOGNITION:
                url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/ocr_sample.zip"
            else:
                raise NotImplementedError(f"not supported task: {task}")

            network.get_file_from_url(url, temp_dir / "mnist.zip")
            io.unzip(temp_dir / "mnist.zip", temp_dir)

            ds = Dataset.from_coco(
                name=name,
                root_dir=root_dir,
                task=task,
                coco_file=str(temp_dir / "coco.json"),
                coco_root_dir=str(temp_dir / "images"),
            )
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(temp_dir)

        ds.create_index()
        return ds

    @classmethod
    def get_dataset_list(cls, root_dir: str = None) -> list[str]:
        """
        Get dataset name list in root_dir.

        Args:
            root_dir (str, optional): dataset root directory. Defaults to None.

        Returns:
            list[str]: dataset name list.
        """
        root_dir = Dataset.parse_root_dir(root_dir)

        if not root_dir.exists():
            return []

        dataset_name_list = []
        for dataset_dir in root_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_info_file = dataset_dir / Dataset.DATASET_INFO_FILE_NAME
                if dataset_info_file.exists():
                    dataset_name_list.append(dataset_dir.name)
        return dataset_name_list

    def _initialize(self):
        """Initialize Dataset.
        It creates necessary directories under {dataset_root_dir}/{dataset_name}.
        """

        if self.initialized():
            raise FileExistsError(f"{self.name} is already initialized.")

        io.make_directory(self.raw_image_dir)
        io.make_directory(self.image_dir)
        io.make_directory(self.annotation_dir)
        io.make_directory(self.category_dir)

    def initialized(self) -> bool:
        """Check if Dataset has been initialized or not.

        Returns:
            bool:
                initialized -> True
                not initialized -> False
        """
        return self.dataset_info_file.exists()

    def save_dataset_info(self):
        """Save DatasetInfo."""
        DatasetInfo(
            name=self.name,
            task=self.task,
            categories=list(map(lambda x: x.to_dict(), self.categories)),
            created=self.created,
        ).save_yaml(self.dataset_info_file)

    def trainable(self) -> bool:
        """Check if Dataset is trainable or not.

        Returns:
            bool:
                trainable -> True
                not trainable -> False
        """
        num_images_per_category: dict[int, int] = self.get_num_images_per_category()
        for category_id, image_num in num_images_per_category.items():
            if image_num < Dataset.MINIMUM_TRAINABLE_IMAGE_NUM_PER_CATEGORY:
                return False
        return True

    def _check_trainable(self):
        """
        Check if Dataset is trainable or not.

        Raises:
            ValueError: if dataset has not enough annotations.
        """
        if not self.trainable():
            raise ValueError(
                "Dataset is not trainable\n"
                + f"Please check if the MINIMUM_TRAINABLE_IMAGE_NUM_PER_CATEGORY={Dataset.MINIMUM_TRAINABLE_IMAGE_NUM_PER_CATEGORY} is satisfied\n"
                + "Your dataset is consisted of\n"
                + "\n".join(
                    [
                        f"  - {category.name}: {self.get_num_images_per_category()[category.category_id]} images"
                        for category in self.get_categories()
                    ]
                )
            )

    def get_dataset_info(self) -> DatasetInfo:
        """Get DatasetInfo.

        Returns:
            DatasetInfo: DatasetInfo
        """
        dataset_info = DatasetInfo.load(self.dataset_info_file)
        if not hasattr(dataset_info, "categories"):
            dataset_info.categories = self.get_categories()
            self.save_dataset_info()
        return dataset_info

    # get
    def get_images(self, image_ids: list[int] = None, labeled: bool = True) -> list[Image]:
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
        return sorted(
            [
                Category.from_json(f, self.task)
                for f in (
                    [self.category_dir / f"{category_id}.json" for category_id in category_ids]
                    if category_ids
                    else self.category_dir.glob("*.json")
                )
            ],
            key=lambda x: x.category_id,
        )

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
            return [Annotation.from_json(f, self.task) for f in self.annotation_dir.glob("*/*.json")]

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
            return [Annotation.from_json(f, self.task) for f in self.prediction_dir.glob("*/*.json")]

    def get_num_images_per_category(self) -> dict[int, int]:
        self.num_images_per_category = {
            category_id: len(images) for category_id, images in self.category_to_images.items()
        }
        return self.num_images_per_category

    def get_num_annotations_per_category(self) -> dict[int, int]:
        num_annotations_per_category = {
            category_id: len(annotations)
            for category_id, annotations in self.category_to_annotations.items()
        }
        return num_annotations_per_category

    def create_index(self):
        """Create index for faster search."""
        self._image_dict = OrderedDict()
        self._unlabeled_image_dict = OrderedDict()
        self._annotation_dict = OrderedDict()
        self._prediction_dict = OrderedDict()
        self._category_dict = OrderedDict()
        self._image_to_annotations = OrderedDict()
        self._image_to_predictions = OrderedDict()
        self._annotation_to_image = OrderedDict()
        self._prediction_to_image = OrderedDict()
        self._category_to_images = OrderedDict()
        self._category_to_unique_images = OrderedDict()
        self._category_name_to_category = OrderedDict()
        self._category_to_annotations = OrderedDict()
        self._category_to_predictions = OrderedDict()

        start = time.time()
        logger.info("Creating index for faster search")

        for image in self.get_images():
            self._image_dict[image.image_id] = image  # image_id: image
            self._image_to_annotations[image.image_id] = []
            self._image_to_predictions[image.image_id] = []

        for annotation in self.get_annotations():
            self._annotation_dict[annotation.annotation_id] = annotation  # annotation_id: annotation
            self._image_to_annotations[annotation.image_id].append(
                annotation
            )  # image_id: [annotations]
            self._annotation_to_image[annotation.annotation_id] = self._image_dict[
                annotation.image_id
            ]  # annotation_id: image

        for prediction in self.get_predictions():
            self._prediction_dict[prediction.annotation_id] = prediction  # annotation_id: prediction
            self._image_to_predictions[prediction.image_id].append(
                prediction
            )  # image_id: [predictions]
            self._prediction_to_image[prediction.annotation_id] = self._image_dict[
                prediction.image_id
            ]  # prediction_id: image

        for category in self.get_categories():
            self._category_dict[category.category_id] = category  # category_id: category
            self._category_name_to_category[category.name] = category  # category_name: category
            self._category_to_unique_images[category.category_id] = []  # category_id: image
            self._category_to_images[category.category_id] = set()
            self._category_to_annotations[category.category_id] = []

        for annotation in self._annotation_dict.values():
            if self.task == TaskType.TEXT_RECOGNITION:
                chars = set(annotation.caption)
                for char in chars:
                    category_id = self._category_name_to_category[char].category_id
                    self._category_to_annotations[category_id].append(annotation)
                    self._category_to_images[category_id].add(
                        self._annotation_to_image[annotation.annotation_id]
                    )
            else:
                self._category_to_annotations[annotation.category_id].append(
                    annotation
                )  # category_id: [annotations]
                self._category_to_images[annotation.category_id].add(
                    self._annotation_to_image[annotation.annotation_id]
                )  # category_id: {images}

        for image_id, annotations in self._image_to_annotations.items():
            if self.task == TaskType.TEXT_RECOGNITION:
                most_common_category = Counter(
                    sum([list(annotation.caption) for annotation in annotations], [])
                ).most_common(1)[0][0]
                most_common_category_id = self._category_name_to_category[
                    most_common_category
                ].category_id
                self._category_to_unique_images[most_common_category_id].append(
                    self._image_dict[image_id]
                )
            else:
                most_common_category_id = Counter(
                    [annotation.category_id for annotation in annotations]
                ).most_common(1)[0][0]
                self._category_to_unique_images[most_common_category_id].append(
                    self._image_dict[image_id]
                )

        for category_id, images in self._category_to_images.items():
            self._category_to_images[category_id] = list(images)

        for prediction in self._prediction_dict.values():
            self._category_to_predictions[prediction.category_id].append(
                prediction
            )  # category_id: [predictions]

        for image in self.get_images(labeled=False):
            self._unlabeled_image_dict[image.image_id] = image  # unlabeled_image_id: image

        logger.info(f"Creating index done {time.time() - start:.2f} seconds")

    # add
    def add_images(self, images: Union[Image, list[Image]]):
        """Add "Image"s to dataset.

        Args:
            images (Union[Image, list[Image]]): list of "Image"s
        """
        if not isinstance(images, list):
            images = [images]

        for item in images:
            item_id = item.image_id
            item_path = self.image_dir / f"{item_id}.json"
            io.save_json(item.to_dict(), item_path)

    def add_categories(self, categories: Union[Category, list[Category]]):
        """Add "Category"s to dataset.

        Args:
            categories (Union[Category, list[Category]]): list of "Category"s
        """
        if not isinstance(categories, list):
            categories = [categories]

        category_names_list = [category.name for category in categories]
        category_names = set(category_names_list)
        if (
            len(category_names) != len(category_names_list)
            or set(self.get_category_names()) & category_names
        ):
            raise ValueError("Category names should be unique")

        for item in categories:
            item_id = item.category_id
            item_path = self.category_dir / f"{item_id}.json"
            io.save_json(item.to_dict(), item_path)

        self.save_dataset_info()

    def add_annotations(self, annotations: Union[Annotation, list[Annotation]]):
        """Add "Annotation"s to dataset.

        Args:
            annotations (Union[Annotation, list[Annotation]]): list of "Annotation"s
        """
        if not isinstance(annotations, list):
            annotations = [annotations]

        categories = self.get_category_names()
        for item in annotations:
            if self.task == TaskType.TEXT_RECOGNITION:
                for char in item.caption:
                    if char not in categories:
                        raise ValueError(f"Category '{char}' is not in dataset")
            item_path = self.annotation_dir / f"{item.image_id}" / f"{item.annotation_id}.json"
            io.save_json(item.to_dict(), item_path, create_directory=True)

    def add_predictions(self, predictions: Union[Annotation, list[Annotation]]):
        """Add "Annotation"s to dataset.

        Args:
            annotations (Union[Annotation, list[Annotation]]): list of "Annotation"s
        """
        if not isinstance(predictions, list):
            predictions = [predictions]

        categories = self.get_category_names()
        for item in predictions:
            if self.task == TaskType.TEXT_RECOGNITION:
                for char in item.caption:
                    if char not in categories:
                        raise ValueError(f"Category '{char}' is not in dataset")
            item_path = self.prediction_dir / f"{item.image_id}" / f"{item.annotation_id}.json"
            io.save_json(item.to_dict(), item_path, create_directory=True)

    # functions
    def split(
        self,
        train_ratio: float,
        val_ratio: float = 0.0,
        test_ratio: float = 0.0,
        method: Union[str, SplitMethod] = SplitMethod.RANDOM,
        seed: int = 0,
    ):
        """
        Split Dataset to train, validation, test, (unlabeled) sets.

        Args:
            train_ratio (float): train num ratio (0 ~ 1).
            val_ratio (float, optional): val num ratio (0 ~ 1).
            test_ratio (float, optional): test num ratio (0 ~ 1).
            method (Union[str, SplitMethod], optional): split method. Defaults to SplitMethod.RANDOM.
            seed (int, optional): random seed. Defaults to 0.

        Raises:
            ValueError: if train_ratio is not between 0.0 and 1.0.
            ValueError: if train_ratio + val_ratio + test_ratio is not 1.0.

        Examples:
            >>> dataset = Dataset.load("some_dataset")
            >>> dataset.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
            >>> dataset.get_split_ids()
            [[1, 2, 3, 4, 5, 6, 7, 8], [9], [10], []]  # train, val, test, unlabeled image ids
        """

        self._check_trainable()

        if train_ratio <= 0.0 or train_ratio >= 1.0:
            raise ValueError(
                "train_ratio must be between 0.0 and 1.0\n" f"given train_ratio: {train_ratio}"
            )

        if val_ratio == 0.0 and test_ratio == 0.0:
            val_ratio = 1 - train_ratio

        if train_ratio + val_ratio + test_ratio != 1.0:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must be 1.0\n"
                f"given train_ratio: {train_ratio}, val_ratio: {val_ratio}, test_ratio: {test_ratio}"
            )

        if method == SplitMethod.RANDOM:
            random.seed(seed)

            train_ids = []
            val_ids = []
            test_ids = []

            for category_id, images in self.category_to_unique_images.items():
                image_num = len(images)
                image_ids = list(map(lambda x: x.image_id, images))
                random.shuffle(image_ids)

                # flatten images to one list [cat]
                train_num = max(int(image_num * train_ratio), 1)
                val_num = max(int(image_num * val_ratio), 1)

                if test_ratio == 0.0:
                    train_ids.extend(image_ids[:train_num])
                    val_ids.extend(image_ids[train_num:])
                else:
                    train_ids.extend(image_ids[:train_num])
                    val_ids.extend(image_ids[train_num : train_num + val_num])
                    test_ids.extend(image_ids[train_num + val_num :])

        else:
            raise ValueError(f"Unknown split method: {method}")

        logger.info(
            f"train num: {len(train_ids)}  val num: {len(val_ids)}  test num: {len(test_ids)}"
        )

        io.save_json(
            train_ids,
            self.train_set_file,
            create_directory=True,
        )
        io.save_json(
            val_ids,
            self.val_set_file,
            create_directory=True,
        )
        io.save_json(
            test_ids,
            self.test_set_file,
            create_directory=True,
        )

        unlabeled_ids = [img.image_id for img in self.get_images(labeled=False)]

        io.save_json(
            unlabeled_ids,
            self.unlabeled_set_file,
            create_directory=True,
        )

    # export
    def get_split_ids(self) -> list[list[int]]:
        """
        Get split ids

        Returns:
            list[list[int]]: split ids
        """
        if not self.train_set_file.exists():
            raise FileNotFoundError("There is no set files. Please run ds.split() first")

        train_ids: list[int] = (
            io.load_json(self.train_set_file) if self.train_set_file.exists() else []
        )
        val_ids: list[int] = io.load_json(self.val_set_file) if self.val_set_file.exists() else []
        test_ids: list[int] = io.load_json(self.test_set_file) if self.test_set_file.exists() else []
        unlabeled_ids: list[int] = (
            io.load_json(self.unlabeled_set_file) if self.unlabeled_set_file.exists() else []
        )

        return [train_ids, val_ids, test_ids, unlabeled_ids]

    def export(self, data_type: Union[str, DataType]) -> str:
        """
        Export Dataset to Specific data formats

        Args:
            data_type (Union[str, DataType]): export data type. one of ["YOLO", "COCO"].

        Raises:
            ValueError: if data_type is not one of DataType.

        Examples:
            >>> dataset = Dataset.load("some_dataset")
            >>> dataset.export(data_type="YOLO")
            path/to/dataset_dir/exports/yolo

            # You can train with exported dataset
            >>> hub.train("path/to/dataset_dir/exports/yolo", ...)

        Returns:
            str: exported dataset directory
        """

        self._check_trainable()

        export_dir: Path = self.export_dir / EXPORT_MAP[data_type.upper()]
        if data_type in [DataType.YOLO, DataType.ULTRALYTICS]:
            export_function = export_yolo
        elif data_type in [DataType.COCO]:
            export_function = export_coco
        elif data_type in [DataType.AUTOCARE_DLT]:
            export_function = export_autocare_dlt
        elif data_type in [DataType.TRANSFORMERS]:
            export_function = export_transformers

        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        try:
            if export_dir.exists():
                io.remove_directory(export_dir)
                warnings.warn(f"{export_dir} already exists. Removing exist export and override.")

            export_dir = export_function(self, export_dir)

            return export_dir

        except Exception as e:
            if export_dir.exists():
                io.remove_directory(export_dir)
            raise e

    def delete(self):
        """Delete Dataset"""
        io.remove_directory(self.dataset_dir)
        del self

    def draw_annotations(self, image_ids=None):
        """
        Draw annotations on images
        Save drawn images to draw_dir

        Args:
            image_ids (list[int], optional): image ids to draw. Defaults to None.

        """
        if not self.draw_dir.exists():
            self.draw_dir.mkdir(parents=True)

        images = self.get_images(image_ids)
        names = self.get_category_names()

        for image in tqdm.tqdm(images):
            np_image = load_image(self.raw_image_dir / image.file_name)
            annotations = self.get_annotations(image.image_id)
            drawn_image = draw_results(np_image, annotations, names)
            save_image(self.draw_dir / image.file_name, drawn_image)
