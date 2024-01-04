from pathlib import Path

import pytest
from waffle_utils.file import io, network

from waffle_hub.dataset import Dataset
from waffle_hub.type import TaskType


@pytest.fixture(scope="session")
def coco_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist.zip"

    tmpdir = tmp_path_factory.mktemp("coco")
    zip_file = tmpdir / "mnist.zip"
    coco_path = tmpdir / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, coco_path, create_directory=True)

    return Path(coco_path)


@pytest.fixture(scope="session")
def yolo_classification_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_yolo_classification.zip"

    tmpdir = tmp_path_factory.mktemp("yolo")
    zip_file = tmpdir / "mnist_yolo_classification.zip"
    yolo_path = tmpdir / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, yolo_path, create_directory=True)

    return Path(yolo_path)


@pytest.fixture(scope="session")
def yolo_object_detection_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_yolo_object_detection.zip"

    tmpdir = tmp_path_factory.mktemp("yolo")
    zip_file = tmpdir / "mnist_yolo_object_detection.zip"
    yolo_path = tmpdir / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, yolo_path, create_directory=True)

    return Path(yolo_path)


@pytest.fixture(scope="session")
def yolo_instance_segmentation_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_yolo_instance_segmentation.zip"

    tmpdir = tmp_path_factory.mktemp("yolo")
    zip_file = tmpdir / "mnist_yolo_instance_segmentation.zip"
    yolo_path = tmpdir / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, yolo_path, create_directory=True)

    return Path(yolo_path)


@pytest.fixture(scope="session")
def transformers_detection_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_huggingface_detection.zip"

    tmpdir = tmp_path_factory.mktemp("transformers")
    zip_file = tmpdir / "mnist_huggingface_detection.zip"
    transformers_path = tmpdir / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, transformers_path, create_directory=True)

    return Path(transformers_path)


@pytest.fixture(scope="session")
def transformers_classification_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_huggingface_classification.zip"

    tmpdir = tmp_path_factory.mktemp("transformers")
    zip_file = tmpdir / "mnist_huggingface_classification.zip"
    transformers_path = tmpdir / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, transformers_path, create_directory=True)

    return Path(transformers_path)


@pytest.fixture(scope="session")
def test_video_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://github.com/snuailab/assets/raw/main/waffle/sample_dataset/video.mp4"

    tmpdir = tmp_path_factory.mktemp("test_video")
    video_path = tmpdir / "video.mp4"

    network.get_file_from_url(url, video_path, create_directory=True)

    return Path(video_path)


@pytest.fixture
def instance_segmentation_dataset(coco_path: Path, tmpdir: Path):
    dataset: Dataset = Dataset.from_coco(
        name="ins_seg",
        task=TaskType.INSTANCE_SEGMENTATION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    dataset.split(0.2, 0.8)

    return dataset


@pytest.fixture
def object_detection_dataset(coco_path: Path, tmpdir: Path):
    dataset: Dataset = Dataset.from_coco(
        name="od",
        task=str(TaskType.OBJECT_DETECTION.value),
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    dataset.split(0.2, 0.8)

    return dataset


@pytest.fixture
def classification_dataset(coco_path: Path, tmpdir: Path):
    dataset: Dataset = Dataset.from_coco(
        name="cls",
        task=str(TaskType.CLASSIFICATION.value),
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    dataset.split(0.2, 0.8)

    return dataset


@pytest.fixture
def text_recognition_dataset(coco_path: Path, tmpdir: Path):
    dataset: Dataset = Dataset.from_coco(
        name="ocr",
        task=str(TaskType.TEXT_RECOGNITION.value),
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    dataset.split(0.8)

    return dataset


@pytest.fixture
def semantic_segmentation_dataset(coco_path: Path, tmpdir: Path):
    dataset: Dataset = Dataset.from_autocare_dlt(
        name="sem_seg",
        task=TaskType.SEMANTIC_SEGMENTATION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    dataset.split(0.2, 0.8)

    return dataset


import torch

from waffle_hub.hub.model.wrapper import ModelWrapper


def _preprocess(x, *args, **kwargs):
    return x


def _postprocess(x, *args, **kwargs):
    return x


@pytest.fixture
def cls_test_model():
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.ReLU(),
        torch.nn.Linear(1, 1),
    )

    model = ModelWrapper(
        model=nn_model,
        preprocess=_preprocess,
        postprocess=_postprocess,
        task=TaskType.CLASSIFICATION,
        categories=["0", "1"],
    )

    return model


@pytest.fixture
def od_test_model():
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.ReLU(),
        torch.nn.Linear(1, 1),
    )

    model = ModelWrapper(
        model=nn_model,
        preprocess=_preprocess,
        postprocess=_postprocess,
        task=TaskType.OBJECT_DETECTION,
        categories=["0", "1"],
    )

    return model


@pytest.fixture
def ins_seg_test_model():
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.ReLU(),
        torch.nn.Linear(1, 1),
    )

    model = ModelWrapper(
        model=nn_model,
        preprocess=_preprocess,
        postprocess=_postprocess,
        task=TaskType.INSTANCE_SEGMENTATION,
        categories=["0", "1"],
    )

    return model


@pytest.fixture
def sem_seg_test_model():
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.ReLU(),
        torch.nn.Linear(1, 1),
    )

    model = ModelWrapper(
        model=nn_model,
        preprocess=_preprocess,
        postprocess=_postprocess,
        task=TaskType.SEMANTIC_SEGMENTATION,
        categories=["0", "1"],
    )

    return model
