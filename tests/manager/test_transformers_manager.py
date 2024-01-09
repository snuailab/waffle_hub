from pathlib import Path

import pytest
from waffle_utils.file import io

from waffle_hub import TrainStatus
from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.manager.adapter.transformers.transformers import TransformersManager
from waffle_hub.type import TaskType


def _train(manager: TransformersManager, dataset: Dataset, image_size: list[int]):
    _delete_artifacts(manager)
    result = manager.train(
        dataset=dataset,
        epochs=1,
        batch_size=4,
        image_size=image_size,
        device="cpu",
        workers=0,
    )
    assert manager.state.status == TrainStatus.SUCCESS
    assert manager.train_config_file.exists()
    assert manager.result == result
    assert manager.weights_dir.exists()
    assert manager.artifacts_dir.exists()
    assert manager.metric_file.exists()

    model = manager.get_model()


def _check_invaild_dataset(
    manager: TransformersManager, wrong_task_dataset: Dataset, image_size: list[int]
):
    _delete_artifacts(manager)
    # wrong task dataset
    with pytest.raises(ValueError):
        manager.train(
            dataset=wrong_task_dataset,
            epochs=1,
            batch_size=4,
            image_size=image_size,
            device="cpu",
            workers=0,
        )
    assert manager.state.status == TrainStatus.FAILED


def _delete_artifacts(manager: TransformersManager):
    manager.delete_artifacts()
    assert manager.state.status == TrainStatus.INIT

    assert not manager.train_config_file.exists()
    assert not manager.artifacts_dir.exists()
    assert not manager.weights_dir.exists()
    assert not manager.train_log_dir.exists()
    assert not manager.metric_file.exists()
    with pytest.raises(ValueError):
        manager.check_train_sanity()


def _test_model_info(task: TaskType):
    # for all available model types and sizes but it takes too long
    # so, we only test one model type and size
    model_type = list(TransformersManager.MODEL_TYPES[task].keys())[0]
    model_size = list(TransformersManager.MODEL_TYPES[task][model_type].keys())[0]

    return model_type, model_size


def test_cls_transformers_manager(
    classification_dataset: Dataset, object_detection_dataset: Dataset, tmpdir: Path
):
    task = TaskType.CLASSIFICATION
    dataset = classification_dataset
    wrong_task_dataset = object_detection_dataset

    model_type, model_size = _test_model_info(task)
    image_size = TransformersManager.DEFAULT_PARAMS[task][model_type][model_size].image_size

    temp_dir = tmpdir / f"{TransformersManager.BACKEND_NAME}_{task}_{model_type}_{model_size}"
    manager = TransformersManager(
        root_dir=temp_dir,
        name="test",
        task=task,
        model_type=model_type,
        model_size=model_size,
    )
    assert manager.state.status == TrainStatus.INIT

    _check_invaild_dataset(manager, wrong_task_dataset, image_size)
    _train(manager, dataset, image_size)

    manager.load(temp_dir)
    manager.delete_manager()
    assert not temp_dir.exists()


def test_od_transformers_manager(
    classification_dataset: Dataset, object_detection_dataset: Dataset, tmpdir: Path
):
    task = TaskType.OBJECT_DETECTION
    dataset = object_detection_dataset
    wrong_task_dataset = classification_dataset

    model_type, model_size = _test_model_info(task)
    image_size = [32, 32]
    temp_dir = tmpdir / f"{TransformersManager.BACKEND_NAME}_{task}_{model_type}_{model_size}"
    manager = TransformersManager(
        root_dir=temp_dir,
        name="test",
        task=task,
        model_type=model_type,
        model_size=model_size,
    )
    assert manager.state.status == TrainStatus.INIT

    _check_invaild_dataset(manager, wrong_task_dataset, image_size)
    _train(manager, dataset, image_size)

    manager.load(temp_dir)
    manager.delete_manager()
    assert not temp_dir.exists()
