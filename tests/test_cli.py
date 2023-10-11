import subprocess
from pathlib import Path

import pytest

from waffle_hub import TaskType
from waffle_hub.dataset import Dataset


def run_cli(cmd):
    ret = subprocess.run(cmd, check=True, shell=True)
    return ret


# hub test
def _new(hub_name: str, tmpdir: Path, task: TaskType):
    cmd = f"python -m waffle_hub.hub.cli new \
        --backend ultralytics \
        --root-dir {tmpdir} \
        --name {hub_name} \
        --task {task} \
        --model-type yolov8 \
        --model-size n \
        --categories [1,2] \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / hub_name).exists()


def _train(hub_name: str, dataset: Dataset, tmpdir: Path):
    cmd = f"python -m waffle_hub.hub.cli train \
        --root-dir {tmpdir} \
        --name {hub_name} \
        --dataset {dataset.name} \
        --dataset_root_dir {dataset.root_dir} \
        --epochs 1 \
        --batch-size 4 \
        --image-size 64 \
        --device cpu \
        --workers 0 \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / hub_name / "artifacts").exists()


def _delete_artifact(hub_name: str, tmpdir: Path):
    cmd = f"python -m waffle_hub.hub.cli delete_artifact \
        --name  {hub_name} \
        --root-dir {tmpdir} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert not (tmpdir / hub_name / "artifacts").exists()


def _train_advance_params(hub_name: str, dataset: Dataset, tmpdir: Path):
    cmd = (
        f"python -m waffle_hub.hub.cli train \
        --root-dir {tmpdir} \
        --name {hub_name} \
        --dataset {dataset.name} \
        --dataset_root_dir {dataset.root_dir} \
        --epochs 1 \
        --batch-size 4 \
        --image-size 64 \
        --device cpu \
        --workers 0 \
    "
        + ' --advance_params "{box: 3}"'
    )
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / hub_name / "artifacts").exists()


def _inference(hub_name: str, dataset: Dataset, tmpdir: Path):
    cmd = f"python -m waffle_hub.hub.cli inference \
        --root-dir {tmpdir} \
        --name {hub_name} \
        --source {dataset.raw_image_dir} \
        --confidence-threshold 0.25 \
        --device cpu \
        --workers 0 \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / hub_name / "inferences").exists()


def _evaluate(hub_name: str, dataset: Dataset, tmpdir: Path):
    cmd = f"python -m waffle_hub.hub.cli evaluate \
        --name {hub_name} \
        --root-dir {tmpdir} \
        --dataset {dataset.name} \
        --dataset_root_dir {dataset.root_dir} \
        --set-name test \
        --device cpu \
        --workers 0 \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / hub_name / "evaluate.json").exists()


def _export_onnx(hub_name: str, tmpdir: Path):
    cmd = f"python -m waffle_hub.hub.cli export_onnx \
        --name {hub_name} \
        --root-dir {tmpdir} \
        --device cpu \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / hub_name / "weights" / "model.onnx").exists()


def _export_waffle(hub_name: str, tmpdir: Path):
    cmd = f"python -m waffle_hub.hub.cli export_waffle \
        --name {hub_name} \
        --root-dir {tmpdir} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / hub_name / f"{hub_name}.waffle").exists()


def _from_waffle_file(hub_name: str, dataset: Dataset, tmpdir: Path):
    cmd = f'python -m waffle_hub.hub.cli from_waffle_file \
        --name from_waffle_file_test \
        --waffle_file {tmpdir / hub_name / f"{hub_name}.waffle"} \
        --root-dir {tmpdir} \
    '
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / hub_name).exists()

    # test_from_waffle_file_inference
    cmd = f"python -m waffle_hub.hub.cli inference \
        --root-dir {tmpdir} \
        --name from_waffle_file_test \
        --source {dataset.raw_image_dir} \
        --confidence-threshold 0.25 \
        --device cpu \
        --workers 0 \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / "from_waffle_file_test" / "inferences").exists()


def _delete_hub(hub_name: str, tmpdir: Path):
    cmd = f"python -m waffle_hub.hub.cli delete_hub \
        --name  {hub_name} \
        --root-dir {tmpdir} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert not (tmpdir / hub_name).exists()


def test_hub(tmpdir: Path, object_detection_dataset: Dataset):
    hub_name = "test_hub"
    dataset = object_detection_dataset
    _new(hub_name, tmpdir, TaskType.OBJECT_DETECTION)
    _train(hub_name, dataset, tmpdir)
    _inference(hub_name, dataset, tmpdir)
    _evaluate(hub_name, dataset, tmpdir)
    _export_onnx(hub_name, tmpdir)
    _export_waffle(hub_name, tmpdir)
    _from_waffle_file(hub_name, dataset, tmpdir)
    _delete_hub(hub_name, tmpdir)


# dataset
def _split(dataset_name: str, tmpdir: Path):
    cmd = f"python -m waffle_hub.dataset.cli split \
        --name {dataset_name} \
        --root-dir {tmpdir} \
        --train-ratio 0.8 \
        --val-ratio 0.1 \
        --test-ratio 0.1 \
        --method random \
        --seed 42 \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / dataset_name / "sets" / "train.json").exists()


def _export(dataset_name: str, tmpdir: Path):
    for data_type in ["ULTRALYTICS", "COCO"]:
        cmd = f"python -m waffle_hub.dataset.cli export \
            --data-type {data_type} \
            --name {dataset_name} \
            --root-dir {tmpdir} \
        "
        ret = run_cli(cmd)
        assert ret.returncode == 0
        assert (tmpdir / dataset_name / "exports" / data_type).exists()


@pytest.mark.parametrize("task", [TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION])
def test_dataset_coco(coco_path: Path, tmpdir: Path, task: TaskType):
    dataset_name = "from_coco"
    cmd = f"python -m waffle_hub.dataset.cli from_coco \
        --name {dataset_name} \
        --root-dir {tmpdir} \
        --coco-file {coco_path / 'coco.json'} \
        --coco-root-dir {coco_path / 'images'} \
        --task {task} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / dataset_name).exists()
    _split(dataset_name, tmpdir)
    _export(dataset_name, tmpdir)


def test_dataset_yolo_cls(yolo_classification_path: Path, tmpdir: Path):
    dataset_name = "from_yolo"
    cmd = f"python -m waffle_hub.dataset.cli from_yolo \
        --name {dataset_name} \
        --root-dir {tmpdir} \
        --task classification \
        --yolo-root-dir {yolo_classification_path} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / dataset_name).exists()
    _split(dataset_name, tmpdir)
    _export(dataset_name, tmpdir)


def test_dataset_yolo_obj(yolo_object_detection_path: Path, tmpdir: Path):
    dataset_name = "from_yolo"
    cmd = f"python -m waffle_hub.dataset.cli from_yolo \
        --name {dataset_name} \
        --root-dir {tmpdir} \
        --task object_detection \
        --yolo-root-dir {yolo_object_detection_path} \
        --yaml-path {yolo_object_detection_path / 'data.yaml'} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / dataset_name).exists()
    _split(dataset_name, tmpdir)
    _export(dataset_name, tmpdir)


def test_dataset_transformer_cls(transformers_classification_path: Path, tmpdir: Path):
    dataset_name = "from_hf"
    cmd = f"python -m waffle_hub.dataset.cli from_transformers \
        --name {dataset_name} \
        --root-dir {tmpdir} \
        --task classification \
        --dataset-dir {transformers_classification_path} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / dataset_name).exists()
    _split(dataset_name, tmpdir)
    _export(dataset_name, tmpdir)


def test_dataset_transformer_obj(transformers_detection_path: Path, tmpdir: Path):
    dataset_name = "from_hf"
    cmd = f"python -m waffle_hub.dataset.cli from_transformers \
        --name {dataset_name} \
        --root-dir {tmpdir} \
        --task object_detection \
        --dataset-dir {transformers_detection_path} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / dataset_name).exists()
    _split(dataset_name, tmpdir)
    _export(dataset_name, tmpdir)


def test_dataset_clone(object_detection_dataset: Dataset, tmpdir: Path):
    dataset = object_detection_dataset
    cmd = f"python -m waffle_hub.dataset.cli clone \
        --src-name {dataset.name} \
        --src-root-dir {dataset.root_dir} \
        --name clone \
        --root-dir {tmpdir} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / "clone").exists()


def test_dataset_delete(object_detection_dataset: Dataset):
    dataset = object_detection_dataset
    dataset_path = dataset.dataset_dir
    cmd = f"python -m waffle_hub.dataset.cli delete \
        --name {dataset.name} \
        --root-dir {dataset.root_dir} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert not (dataset_path).exists()


def test_dataset_get_split_ids(object_detection_dataset: Dataset):
    dataset = object_detection_dataset
    cmd = f"python -m waffle_hub.dataset.cli get_split_ids \
        --name {dataset.name} \
        --root-dir {dataset.root_dir} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0


def test_dataset_merge(coco_path: Path, yolo_object_detection_path: Path, tmpdir: Path):
    dataset1_name = "coco"
    dataset2_name = "yolo"
    Dataset.from_coco(
        name=dataset1_name,
        root_dir=tmpdir,
        coco_file=coco_path / "coco.json",
        coco_root_dir=coco_path / "images",
        task=TaskType.OBJECT_DETECTION,
    )
    Dataset.from_yolo(
        name=dataset2_name,
        root_dir=tmpdir,
        task=TaskType.OBJECT_DETECTION,
        yolo_root_dir=yolo_object_detection_path,
        yaml_path=yolo_object_detection_path / "data.yaml",
    )
    cmd = f"python -m waffle_hub.dataset.cli merge \
        --name merge \
        --root-dir {tmpdir} \
        --src-names [{dataset1_name},{dataset2_name}] \
        --src-root-dirs {tmpdir} \
        --task object_detection \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / "merge").exists()


@pytest.mark.parametrize("task", [TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION])
def test_dataset_sample(tmpdir: Path, task: TaskType):
    cmd = f"python -m waffle_hub.dataset.cli sample \
        --name sample \
        --root-dir {tmpdir} \
        --task {task} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
    assert (tmpdir / "sample").exists()


def test_dataset_get_fields(object_detection_dataset: Dataset):
    dataset = object_detection_dataset
    # image
    cmd = f"python -m waffle_hub.dataset.cli get_images \
        --name {dataset.name} \
        --root-dir {dataset.root_dir} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0

    # annotation
    cmd = f"python -m waffle_hub.dataset.cli get_annotations \
        --name {dataset.name} \
        --root-dir {dataset.root_dir} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0

    # category
    cmd = f"python -m waffle_hub.dataset.cli get_categories \
        --name {dataset.name} \
        --root-dir {dataset.root_dir} \
    "
    ret = run_cli(cmd)
    assert ret.returncode == 0
