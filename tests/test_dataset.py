from pathlib import Path

import pytest

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.schema.fields import Annotation, Category, Image
from waffle_hub.utils.data import ImageDataset, LabeledDataset


def test_annotation():

    bbox = [100, 100, 100, 100]
    segmentation = [110, 110, 130, 130, 110, 130]
    keypoints = [0, 0, 0, 130, 130, 1, 110, 130, 2]

    # object detection
    a = Annotation.object_detection(
        annotation_id=1,
        image_id=1,
        category_id=1,
        bbox=bbox,
        area=10000,
    )
    assert not a.is_prediction()

    a.score = 0.4
    assert a.is_prediction()

    d = a.to_dict()
    a = Annotation.from_dict(d)

    # classification
    a = Annotation.classification(
        annotation_id=1,
        image_id=1,
        category_id=1,
    )

    # segmentation
    a = Annotation.segmentation(
        annotation_id=1,
        image_id=1,
        category_id=1,
        segmentation=segmentation,
        area=10000,
    )

    # keypoint detection
    a = Annotation.keypoint_detection(
        annotation_id=1,
        image_id=1,
        category_id=1,
        keypoints=keypoints,
        num_keypoints=3,
    )


def test_image():

    image = Image.new(
        image_id=1,
        file_name="test.jpg",
        width=100,
        height=100,
    )

    d = image.to_dict()
    image = Image.from_dict(d)


def test_category():

    # object detection
    category = Category.object_detection(
        category_id=1,
        name="test",
        supercategory="object",
    )

    d = category.to_dict()
    category = Category.from_dict(d)

    # classification
    category = Category.classification(
        category_id=1,
        name="test",
        supercategory="object",
    )

    # segmentation
    category = Category.segmentation(
        category_id=1,
        name="test",
        supercategory="object",
    )

    # keypoint detection
    category = Category.keypoint_detection(
        category_id=1,
        name="test",
        supercategory="object",
        keypoints=["a", "b", "c"],
        skeleton=[[1, 2], [2, 3]],
    )


def test_dataset(tmpdir):

    dataset: Dataset = Dataset.new(name="test_new", task=TaskType.OBJECT_DETECTION, root_dir=tmpdir)
    assert Path(dataset.dataset_dir).exists()

    dataset.delete()
    assert not Path(dataset.dataset_dir).exists()


def test_dataset_coco(coco_path, tmpdir):

    ds = Dataset.from_coco(
        name="from_coco",
        task=TaskType.OBJECT_DETECTION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    assert ds.dataset_info_file.exists()

    ds = Dataset.from_coco(
        name="import_train_val",
        task=TaskType.OBJECT_DETECTION,
        coco_file=[coco_path / "train.json", coco_path / "val.json"],
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) == 60
    assert len(val_ids) == 20
    assert len(val_ids) == len(test_ids)
    assert len(ds.images) == 80

    ds = Dataset.from_coco(
        name="import_train_val_test",
        task=TaskType.OBJECT_DETECTION,
        coco_file=[coco_path / "train.json", coco_path / "val.json", coco_path / "test.json"],
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) == 60
    assert len(val_ids) == 20
    assert len(test_ids) == 20
    assert len(ds.images) == 100

    ds = Dataset.load("from_coco", root_dir=tmpdir)

    ds = Dataset.clone(
        src_name="from_coco",
        name="clone_coco",
        src_root_dir=tmpdir,
        root_dir=tmpdir,
    )

    with pytest.raises(FileNotFoundError):
        ds.export("coco")

    ds.split(0.8)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) + len(val_ids) == len(ds.images)

    ds.split(0.445446, 0.554554)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) + len(val_ids) == len(ds.images)

    ds.split(0.4, 0.4, 0.2)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(ds.images)

    ds.split(0.99999999999999, 0.0)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(val_ids) == 1 and len(test_ids) == 1

    with pytest.raises(ValueError):
        ds.split(0.0, 0.2)

    with pytest.raises(ValueError):
        ds.split(0.9, 0.2)

    ds.export("coco")


def test_dataset_yolo(yolo_path, tmpdir):

    ds = Dataset.from_yolo(
        name="from_yolo",
        task=TaskType.OBJECT_DETECTION,
        yaml_path=yolo_path / "data.yaml",
        root_dir=tmpdir,
    )
    assert ds.dataset_info_file.exists()

    ds = Dataset.load("from_yolo", root_dir=tmpdir)

    ds = Dataset.clone(
        src_name="from_yolo",
        name="clone_yolo",
        src_root_dir=tmpdir,
        root_dir=tmpdir,
    )

    with pytest.raises(FileNotFoundError):
        ds.export("yolo")

    ds.split(0.8)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) + len(val_ids) == len(ds.images)

    ds.split(0.445446, 0.554554)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) + len(val_ids) == len(ds.images)

    ds.split(0.4, 0.4, 0.2)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(ds.images)

    ds.split(0.99999999999999, 0.0)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(val_ids) == 1 and len(test_ids) == 1

    with pytest.raises(ValueError):
        ds.split(0.0, 0.2)

    with pytest.raises(ValueError):
        ds.split(0.9, 0.2)

    ds.export("yolo")


def test_dataset_huggingface(huggingface_path, tmpdir):

    ds = Dataset.from_huggingface(
        name="from_huggingface",
        task=TaskType.OBJECT_DETECTION,
        dataset_dir=huggingface_path,
        root_dir=tmpdir,
    )
    assert ds.dataset_info_file.exists()

    ds = Dataset.load("from_huggingface", root_dir=tmpdir)

    ds = Dataset.clone(
        src_name="from_huggingface",
        name="clone_huggingface",
        src_root_dir=tmpdir,
        root_dir=tmpdir,
    )

    with pytest.raises(FileNotFoundError):
        ds.export("huggingface")

    ds.split(0.8)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) + len(val_ids) == len(ds.images)

    ds.split(0.445446, 0.554554)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) + len(val_ids) == len(ds.images)

    ds.split(0.4, 0.4, 0.2)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(ds.images)

    ds.split(0.99999999999999, 0.0)
    train_ids, val_ids, test_ids, unlabeled_ids = ds.get_split_ids()
    assert len(val_ids) == 1 and len(test_ids) == 1

    with pytest.raises(ValueError):
        ds.split(0.0, 0.2)

    with pytest.raises(ValueError):
        ds.split(0.9, 0.2)

    ds.export("huggingface")


def test_image_dataloader(coco_path, tmpdir):

    image_dataset = ImageDataset(coco_path / "images", 224)
    assert len(image_dataset) == 100
    image_dataloader = image_dataset.get_dataloader(batch_size=32, num_workers=0)
    assert len(image_dataloader) == 4


def test_labled_dataloader(coco_path, tmpdir):
    ds = Dataset.from_coco(
        name="from_coco",
        task=TaskType.OBJECT_DETECTION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    assert ds.dataset_info_file.exists()

    labeled_dataset = LabeledDataset(ds, 224)
    assert len(labeled_dataset) == 100
    labeled_dataloader = labeled_dataset.get_dataloader(batch_size=32, num_workers=0)
    assert len(labeled_dataloader) == 4

    ds.split(0.8)
    labeled_dataset = LabeledDataset(ds, 224, set_name="train")
    assert len(labeled_dataset) == 80

    image, image_info, annotations = labeled_dataset[0]
    assert hasattr(annotations[0], "bbox")
