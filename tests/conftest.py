from pathlib import Path

import pytest
from waffle_utils.file import io, network

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset


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
def superb_ai_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_superbai_detection.zip"

    tmpdir = tmp_path_factory.mktemp("superb_ai")
    zip_file = tmpdir / "mnist_superbai_detection.zip"
    superb_ai_path = tmpdir / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, superb_ai_path, create_directory=True)

    return Path(superb_ai_path)


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
        name="seg",
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
        task=TaskType.OBJECT_DETECTION,
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
        task=TaskType.CLASSIFICATION,
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
        task=TaskType.TEXT_RECOGNITION,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        root_dir=tmpdir,
    )
    dataset.split(0.8)

    return dataset
