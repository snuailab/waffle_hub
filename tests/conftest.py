from pathlib import Path

import pytest
from waffle_utils.file import io, network


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
def yolo_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_yolo_object_detection_splited.zip"

    tmpdir = tmp_path_factory.mktemp("yolo")
    zip_file = tmpdir / "mnist_yolo_object_detection_splied.zip"
    yolo_path = tmpdir / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, yolo_path, create_directory=True)

    info = io.load_yaml(yolo_path / "data.yaml")
    info["path"] = str(yolo_path)
    info["train"] = info["val"] = info["test"] = "images"
    info["names"] = {0: "1", 1: "2"}
    io.save_yaml(info, yolo_path / "data.yaml")

    return Path(yolo_path)


@pytest.fixture(scope="session")
def huggingface_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_huggingface_detection.zip"

    tmpdir = tmp_path_factory.mktemp("huggingface")
    zip_file = tmpdir / "mnist_huggingface_detection.zip"
    huggingface_path = tmpdir / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, huggingface_path, create_directory=True)

    return Path(huggingface_path)
