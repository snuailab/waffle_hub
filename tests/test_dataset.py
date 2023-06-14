from collections import Counter
from pathlib import Path

import pytest
from waffle_utils.file.io import load_json, save_json

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.schema.fields import Annotation, Category, Image
from waffle_hub.utils.data import ImageDataset, LabeledDataset


def test_annotation():

    bbox = [100, 100, 100, 100]
    segmentation = [[110, 110, 130, 130, 110, 130]]
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

    a = Annotation.object_detection(
        annotation_id=1,
        image_id=1,
        category_id=1,
        bbox=bbox,
    )
    assert a.area == 10000

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
    a = Annotation.semantic_segmentation(
        annotation_id=1,
        image_id=1,
        category_id=1,
        segmentation=segmentation,
        area=10000,
    )
    assert a.bbox == [110, 110, 20, 20]
    assert a.area == 10000

    a = Annotation.instance_segmentation(
        annotation_id=1,
        image_id=1,
        category_id=1,
        segmentation=segmentation,
    )
    assert a.bbox == [110, 110, 20, 20]
    assert a.area == 200

    # keypoint detection
    a = Annotation.keypoint_detection(
        annotation_id=1,
        image_id=1,
        category_id=1,
        keypoints=keypoints,
        num_keypoints=3,
    )

    # text recognition
    a = Annotation.text_recognition(
        annotation_id=1,
        image_id=1,
        caption="1",
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
    category = Category.semantic_segmentation(
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

    # text recognition
    category = Category.text_recognition(
        category_id=1,
        name="test",
        supercategory="object",
    )


def test_dataset(tmpdir):

    dataset: Dataset = Dataset.new(name="test_new", task=TaskType.OBJECT_DETECTION, root_dir=tmpdir)
    assert Path(dataset.dataset_dir).exists()

    dataset.delete()
    assert not Path(dataset.dataset_dir).exists()


def _load(dataset_name, root_dir):
    Dataset.load(dataset_name, root_dir=root_dir)


def _clone(dataset_name, root_dir):
    Dataset.clone(
        src_name=dataset_name,
        name="clone_" + dataset_name,
        src_root_dir=root_dir,
        root_dir=root_dir,
    )


def _split(dataset_name, root_dir):
    dataset = Dataset.load(dataset_name, root_dir=root_dir)

    dataset.split(0.8)
    train_ids, val_ids, test_ids, unlabeled_ids = dataset.get_split_ids()
    assert len(train_ids) + len(val_ids) == len(dataset.images)

    dataset.split(0.445446, 0.554554)
    train_ids, val_ids, test_ids, unlabeled_ids = dataset.get_split_ids()
    assert len(train_ids) + len(val_ids) == len(dataset.images)

    dataset.split(0.4, 0.4, 0.2)
    train_ids, val_ids, test_ids, unlabeled_ids = dataset.get_split_ids()
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(dataset.images)

    dataset.split(0.99999999999999, 0.0)
    train_ids, val_ids, test_ids, unlabeled_ids = dataset.get_split_ids()
    assert len(dataset.categories) == len(val_ids) == len(test_ids)

    dataset.split(0.00000000000001, 0.0)
    train_ids, val_ids, test_ids, unlabeled_ids = dataset.get_split_ids()
    assert len(dataset.categories) == len(train_ids)

    with pytest.raises(ValueError):
        dataset.split(0.0, 0.2)

    with pytest.raises(ValueError):
        dataset.split(0.9, 0.2)


def _export(dataset_name, task: TaskType, root_dir):
    dataset = Dataset.load(dataset_name, root_dir=root_dir)
    dataset.split(0.05)

    if task in [TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION, TaskType.CLASSIFICATION]:
        dataset.export("coco")
    if task in [TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION, TaskType.CLASSIFICATION]:
        dataset.export("yolo")
    if task in [TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION]:
        dataset.export("transformers")
    if task in [TaskType.OBJECT_DETECTION, TaskType.TEXT_RECOGNITION, TaskType.CLASSIFICATION]:
        dataset.export("autocare_dlt")


# test dummy
def _dummy(dataset_name, task: TaskType, image_num, category_num, unlabeled_image_num, root_dir):
    dataset = Dataset.dummy(
        name=dataset_name,
        task=task,
        image_num=image_num,
        category_num=category_num,
        unlabeled_image_num=unlabeled_image_num,
        root_dir=root_dir,
    )
    assert len(dataset.images) == image_num
    assert len(dataset.categories) == category_num
    assert len(dataset.unlabeled_images) == unlabeled_image_num


def _total_dummy(
    dataset_name, task: TaskType, image_num, category_num, unlabeled_image_num, root_dir
):
    _dummy(dataset_name, task, image_num, category_num, unlabeled_image_num, root_dir)
    _load(dataset_name, root_dir)
    _clone(dataset_name, root_dir)
    _split(dataset_name, root_dir)
    _export(dataset_name, task, root_dir)


def test_dummy(tmpdir):
    for task in [
        TaskType.CLASSIFICATION,
        TaskType.OBJECT_DETECTION,
        TaskType.INSTANCE_SEGMENTATION,
        TaskType.TEXT_RECOGNITION,
    ]:
        _total_dummy(f"dummy_{task}", task, 100, 5, 10, tmpdir)

    with pytest.raises(ValueError):
        _total_dummy("dummy", TaskType.CLASSIFICATION, 3, 3, 0, tmpdir)


# test dummy
def _dummy(dataset_name, task: TaskType, image_num, category_num, unlabeled_image_num, root_dir):
    dataset = Dataset.dummy(
        name=dataset_name,
        task=task,
        image_num=image_num,
        category_num=category_num,
        unlabeled_image_num=unlabeled_image_num,
        root_dir=root_dir,
    )
    assert len(dataset.images) == image_num
    assert len(dataset.categories) == category_num
    assert len(dataset.unlabeled_images) == unlabeled_image_num


def _total_dummy(
    dataset_name, task: TaskType, image_num, category_num, unlabeled_image_num, root_dir
):
    _dummy(dataset_name, task, image_num, category_num, unlabeled_image_num, root_dir)
    _load(dataset_name, root_dir)
    _clone(dataset_name, root_dir)
    _split(dataset_name, root_dir)
    _export(dataset_name, task, root_dir)


def test_dummy(tmpdir):
    for task in [TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]:
        _total_dummy(f"dummy_{task}", task, 100, 5, 10, tmpdir)

    with pytest.raises(ValueError):
        _total_dummy("dummy", TaskType.CLASSIFICATION, 3, 3, 0, tmpdir)


# test coco
def _from_coco(dataset_name, task: TaskType, coco_path, root_dir):
    dataset = Dataset.from_coco(
        name=dataset_name,
        task=task,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=root_dir,
    )
    assert dataset.dataset_info_file.exists()

    dataset = Dataset.from_coco(
        name=f"{task}_import_train_val",
        task=task,
        coco_file=[coco_path / "train.json", coco_path / "val.json"],
        coco_root_dir=coco_path / "images",
        root_dir=root_dir,
    )
    train_ids, val_ids, test_ids, unlabeled_ids = dataset.get_split_ids()
    assert len(train_ids) == 60
    assert len(val_ids) == 20
    assert len(val_ids) == len(test_ids)
    assert len(dataset.images) == 80

    dataset = Dataset.from_coco(
        name=f"{task}_import_train_val_test",
        task=task,
        coco_file=[coco_path / "train.json", coco_path / "val.json", coco_path / "test.json"],
        coco_root_dir=coco_path / "images",
        root_dir=root_dir,
    )
    train_ids, val_ids, test_ids, unlabeled_ids = dataset.get_split_ids()
    assert len(train_ids) == 60
    assert len(val_ids) == 20
    assert len(test_ids) == 20
    assert len(dataset.images) == 100


def _total_coco(dataset_name, task: TaskType, coco_path, root_dir):
    _from_coco(dataset_name, task, coco_path, root_dir)
    _load(dataset_name, root_dir)
    _clone(dataset_name, root_dir)
    _split(dataset_name, root_dir)
    _export(dataset_name, task, root_dir)


@pytest.mark.parametrize(
    "task", [TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]
)
def test_coco(coco_path, tmpdir, task):
    _total_coco(f"coco_{task}", task, coco_path, tmpdir)


# test autocare_dlt
def _from_autocare_dlt(dataset_name, task: TaskType, coco_path, root_dir):
    dataset = Dataset.from_autocare_dlt(
        name=dataset_name,
        task=task,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=root_dir,
    )
    assert dataset.dataset_info_file.exists()


def _total_autocare_dlt(dataset_name, task: TaskType, coco_path, root_dir):
    _from_autocare_dlt(dataset_name, task, coco_path, root_dir)
    _load(dataset_name, root_dir)
    _clone(dataset_name, root_dir)
    _split(dataset_name, root_dir)
    _export(dataset_name, task, root_dir)


@pytest.mark.parametrize(
    "task", [TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION, TaskType.TEXT_RECOGNITION]
)
def test_autocare_dlt(coco_path, tmpdir, task):
    _total_autocare_dlt(f"autocare_dlt_{task}", task, coco_path, tmpdir)


# test transformers
def _from_transformers(dataset_name, task: TaskType, transformers_path, root_dir):
    dataset = Dataset.from_transformers(
        name=dataset_name,
        task=task,
        dataset_dir=transformers_path,
        root_dir=root_dir,
    )
    assert dataset.dataset_info_file.exists()

    train_ids, val_ids, test_ids, unlabeled_ids = dataset.get_split_ids()
    assert len(train_ids) == 80
    assert len(dataset.images) == 100


def _total_transformers(dataset_name, task: TaskType, transformers_path, root_dir):
    _from_transformers(dataset_name, task, transformers_path, root_dir)
    _load(dataset_name, root_dir)
    _clone(dataset_name, root_dir)
    _split(dataset_name, root_dir)
    _export(dataset_name, task, root_dir)


def test_transformers(transformers_detection_path, transformers_classification_path, tmpdir):
    _total_transformers(
        "transformers_object_detection",
        TaskType.OBJECT_DETECTION,
        transformers_detection_path,
        tmpdir,
    )
    _total_transformers(
        "transformers_classification",
        TaskType.CLASSIFICATION,
        transformers_classification_path,
        tmpdir,
    )


# dataloader
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
    image, image_info, annotations = labeled_dataset[0]
    assert hasattr(annotations[0], "bbox")


# etc
def test_sample(tmpdir):
    for task_type in TaskType:
        try:
            Dataset.sample(
                name=f"sample_{task_type}",
                root_dir=tmpdir,
                task=task_type,
            )
        except NotImplementedError:
            continue

        assert (tmpdir / f"sample_{task_type}").exists()


def test_merge(coco_path, tmpdir):
    ds1 = Dataset.from_coco(
        name="ds1",
        task=TaskType.OBJECT_DETECTION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    ds2 = Dataset.from_coco(
        name="ds2",
        task=TaskType.OBJECT_DETECTION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )

    cateids_of_ann = [annotation.category_id for annotation in ds1.annotations.values()]
    category_counts = Counter(cateids_of_ann)

    category_1_num = category_counts[1]
    category_2_num = category_counts[2]

    ds = Dataset.merge(
        name="merge",
        src_names=["ds1", "ds2"],
        src_root_dirs=[tmpdir, tmpdir],
        root_dir=tmpdir,
        task=TaskType.OBJECT_DETECTION,
    )

    cateids_of_ann = [annotation.category_id for annotation in ds.annotations.values()]
    category_counts = Counter(cateids_of_ann)

    assert (ds.raw_image_dir).exists()
    assert len(ds.images) == 100
    assert len(ds.annotations) == 100
    assert len(ds.categories) == 2
    assert category_counts[1] == category_1_num
    assert category_counts[2] == category_2_num

    # test merge with different category name
    category = load_json(ds1.category_dir / "1.json")
    category["name"] = "one"
    save_json(category, ds1.category_dir / "1.json")

    ds = Dataset.merge(
        name="merge2",
        src_names=["ds1", "ds2"],
        src_root_dirs=[tmpdir, tmpdir],
        root_dir=tmpdir,
        task=TaskType.OBJECT_DETECTION,
    )

    cateids_of_ann = [annotation.category_id for annotation in ds.annotations.values()]
    category_counts = Counter(cateids_of_ann)

    assert len(ds.images) == 100
    assert len(ds.annotations) == 100 + category_1_num
    assert len(ds.categories) == 3

    assert category_counts[1] == category_1_num
    assert category_counts[2] == category_2_num
    assert category_counts[3] == category_1_num


def test_cached_property(tmpdir):
    ds = Dataset.dummy(
        name="cached_property",
        task=TaskType.CLASSIFICATION,
        image_num=10,
        category_num=1,
        unlabeled_image_num=0,
        root_dir=tmpdir,
    )
    before_img_to_anns = ds.image_to_annotations
    before_cate_to_imgs = ds.category_to_images
    before_cate_to_anns = ds.category_to_annotations
    before_cate_names = ds.category_names
    before_cates = ds.categories
    before_imgs = ds.images
    before_anns = ds.annotations

    # image - unlabeled
    new_img1 = Image.new(11, "11.png", 32, 32)
    new_img2 = Image.new(12, "12.png", 32, 32)
    ds.add_images([new_img1, new_img2])
    assert len(ds.images) == len(before_imgs)
    assert len(ds.image_to_annotations.keys()) == len(before_img_to_anns.keys())

    # annotation / image - labeled
    new_ann = Annotation.classification(11, 11, 1)
    ds.add_annotations([new_ann])
    assert len(ds.annotations) == len(before_anns) + 1
    assert len(ds.images) == len(before_imgs) + 1
    assert len(ds.image_to_annotations.keys()) == len(before_img_to_anns.keys()) + 1

    # category
    new_cate = Category.classification(2, "new")
    ds.add_categories([new_cate])
    assert len(ds.categories) == len(before_cates) + 1
    assert len(ds.category_names) == len(before_cate_names) + 1

    new_ann = Annotation.classification(11, 11, 2)
    ds.add_annotations([new_ann])
    assert len(ds.category_to_annotations.keys()) == len(before_cate_to_anns.keys()) + 1
    assert len(ds.category_to_images.keys()) == len(before_cate_to_imgs.keys()) + 1
