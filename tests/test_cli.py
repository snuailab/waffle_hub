import subprocess
from pathlib import Path

import pytest
from waffle_utils.file.io import unzip
from waffle_utils.file.network import get_file_from_url


def run_cli(cmd):
    ret = subprocess.run(cmd, check=True, shell=True)
    return ret


@pytest.fixture(scope="session")
def test_dir(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("test"))


def test_dataset_new(test_dir: Path):
    cmd = f"python -m waffle_hub.run dataset new \
        --name new \
        --root-dir {test_dir / 'datasets'} \
        --task classification \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0


def test_dataset_from_coco(test_dir: Path):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist.zip"
    coco_dir = test_dir / "datasets" / "mnist_coco"

    get_file_from_url(url, str(test_dir), True)
    unzip(str(test_dir / "mnist.zip"), coco_dir)

    cmd = f"python -m waffle_hub.run dataset from_coco \
        --name from_coco \
        --root-dir {test_dir / 'datasets'} \
        --coco-file {coco_dir / 'coco.json'} \
        --coco-root-dir {coco_dir / 'images'} \
        --task classification \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "datasets" / "from_coco").exists()


def test_dataset_from_yolo(test_dir: Path):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_yolo_object_detection_splited.zip"
    yolo_dir = test_dir / "datasets" / "mnist_yolo"

    get_file_from_url(url, str(test_dir), True)
    unzip(str(test_dir / "mnist_yolo_object_detection_splited.zip"), yolo_dir)

    cmd = f"python -m waffle_hub.run dataset from_yolo \
        --name from_yolo \
        --root-dir {test_dir / 'datasets'} \
        --task object_detection \
        --yaml-path {yolo_dir / 'data.yaml'} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "datasets" / "from_yolo").exists()


def test_dataset_from_huggingface(test_dir: Path):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_huggingface_classification.zip"
    hf_dir = test_dir / "datasets" / "mnist_hf"

    get_file_from_url(url, str(test_dir), True)
    unzip(str(test_dir / "mnist_huggingface_classification.zip"), hf_dir)

    cmd = f"python -m waffle_hub.run dataset from_huggingface \
        --name from_hf \
        --root-dir {test_dir / 'datasets'} \
        --task classification \
        --dataset-dir {hf_dir} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "datasets" / "from_hf").exists()


def test_dataset_split(test_dir: Path):
    cmd = f"python -m waffle_hub.run dataset split \
        --name from_coco \
        --root-dir {test_dir / 'datasets'} \
        --train-ratio 0.8 \
        --val-ratio 0.1 \
        --test-ratio 0.1 \
        --method random \
        --seed 42 \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "datasets" / "from_coco" / "sets" / "train.json").exists()


def test_dataset_export(test_dir: Path):
    cmd = f"python -m waffle_hub.run dataset export \
        --data-type ultralytics \
        --name from_coco \
        --root-dir {test_dir / 'datasets'} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "datasets" / "from_coco" / "exports" / "YOLO").exists()


def test_dataset_clone(test_dir: Path):
    cmd = f"python -m waffle_hub.run dataset clone \
        --src-name from_coco \
        --name clone \
        --src-root-dir {test_dir / 'datasets'} \
        --root-dir {test_dir / 'datasets'} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "datasets" / "clone").exists()


def test_hub_new(test_dir: Path):
    cmd = f'python -m waffle_hub.run hub new \
        --backend ultralytics \
        --root-dir {test_dir / "hubs"} \
        --name test \
        --task classification \
        --model-type yolov8 \
        --model-size n \
        --categories 1 \
        --categories 2 \
    '
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "hubs" / "test").exists()


def test_hub_train(test_dir: Path):
    cmd = f'python -m waffle_hub.run hub train \
        --backend ultralytics \
        --root-dir {test_dir / "hubs"} \
        --name test \
        --dataset-path {test_dir / "datasets" / "from_coco" / "exports" / "YOLO" } \
        --epochs 1 \
        --batch-size 4 \
        --image-size 16 \
        --learning-rate 0.001 \
        --letter-box \
        --device cpu \
        --workers 0 \
        --seed 0 \
        --verbose \
    '
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "hubs" / "test" / "artifacts").exists()


def test_hub_inference(test_dir: Path):
    cmd = f'python -m waffle_hub.run hub inference \
        --backend ultralytics \
        --root-dir {test_dir / "hubs"} \
        --name test \
        --source {test_dir / "datasets" / "mnist" / "exports" / "YOLO" / "test" / "images" } \
        --confidence-threshold 0.25 \
        --device cpu \
        --workers 0 \
    '
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "hubs" / "test" / "inferences").exists()


def test_hub_evaluate(test_dir: Path):
    cmd = f'python -m waffle_hub.run hub evaluate \
        --backend ultralytics \
        --name test \
        --root-dir {test_dir / "hubs"} \
        --dataset-name from_coco \
        --dataset-root-dir {test_dir / "datasets"} \
        --set-name test \
        --device cpu \
        --workers 0 \
    '
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "hubs" / "test" / "evaluate.json").exists()


def test_hub_export(test_dir: Path):
    cmd = f'python -m waffle_hub.run hub export \
        --backend ultralytics \
        --name test \
        --root-dir {test_dir / "hubs"} \
        --device cpu \
    '
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (test_dir / "hubs" / "test" / "weights" / "model.onnx").exists()
