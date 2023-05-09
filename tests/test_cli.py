import subprocess
from pathlib import Path

from waffle_utils.file.io import unzip
from waffle_utils.file.network import get_file_from_url


def run_cli(cmd):
    cmd = cmd.replace("\\", " ")
    ret = subprocess.run(cmd.split(), check=True)
    return ret


def test_dataset_new(tmpdir: Path):
    cmd = f"python -m waffle_hub.run dataset new \
        --name new \
        --root-dir {tmpdir / 'datasets'} \
        --task classification \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0


def test_dataset_from_coco(tmpdir: Path):
    get_file_from_url("url", str(tmpdir), True)
    unzip(str(tmpdir / "mnist.zip"), str(tmpdir / "datasets"))

    cmd = f"python -m waffle_hub.run dataset from_coco \
        --name mnist \
        --root-dir {tmpdir / 'datasets'} \
        --coco-file datasets/mnist.json \
        --coco-root-dir datasets \
        --task classification \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0


def test_dataset_from_yolo(tmpdir: Path):
    pass


def test_dataset_from_huggingface(tmpdir: Path):
    pass


def test_dataset_split(tmpdir: Path):
    pass


def test_dataset_export(tmp_path: Path):
    pass


def test_dataset_clone(tmp_path: Path):
    pass


def test_hub_new(tmp_path: Path):
    pass


def test_hub_train(tmp_path: Path):
    pass


def test_hub_inference(tmp_path: Path):
    pass


def test_hub_evaluate(tmp_path: Path):
    pass


def test_hub_export(tmp_path: Path):
    pass
