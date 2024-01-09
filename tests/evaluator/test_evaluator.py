from pathlib import Path

import pytest
from waffle_utils.file import io

from waffle_hub import EvaluateStatus
from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.evaluator.evaluator import Evaluator
from waffle_hub.hub.model.boring_wrapper import BoringModelWrapper
from waffle_hub.type import TaskType


def _evaluate(task: TaskType, dataset: Dataset, tmpdir: Path):
    model = BoringModelWrapper(task=task, categories=["1", "2"], batch_size=1)
    evaluator = Evaluator(
        root_dir=tmpdir,
        model=model,
    )
    assert evaluator.state.status == EvaluateStatus.INIT

    result = evaluator.evaluate(
        dataset=dataset,
        batch_size=1,
        image_size=[32, 32],
        letter_box=False,
        device="cpu",
        workers=0,
    )
    assert evaluator.state.status == EvaluateStatus.SUCCESS
    assert evaluator.result == result
    assert evaluator.evaluate_file.exists()
    assert io.load_json(evaluator.evaluate_file) == result.eval_metrics


def _check_invaild_dataset(task: TaskType, dataset: Dataset, wrong_dataset: Dataset, tmpdir: Path):
    # wrong category
    with pytest.raises(ValueError):
        model = BoringModelWrapper(task=task, categories=["1", "2", "3", "4"], batch_size=1)
        evaluator = Evaluator(
            root_dir=tmpdir,
            model=model,
        )
        assert evaluator.state.status == EvaluateStatus.INIT
        evaluator.evaluate(
            dataset=dataset,
            batch_size=1,
            image_size=[32, 32],
            letter_box=False,
            device="cpu",
            workers=0,
        )
    assert evaluator.state.status == EvaluateStatus.FAILED

    model = BoringModelWrapper(task=task, categories=["1", "2"], batch_size=1)
    evaluator = Evaluator(
        root_dir=tmpdir,
        model=model,
    )
    assert evaluator.state.status == EvaluateStatus.INIT

    # wrong task dataset
    with pytest.raises(ValueError):
        evaluator.evaluate(
            dataset=wrong_dataset,
            batch_size=1,
            image_size=[32, 32],
            letter_box=False,
            device="cpu",
            workers=0,
        )
    assert evaluator.state.status == EvaluateStatus.FAILED


def test_cls_evaluator(
    classification_dataset: Dataset, object_detection_dataset: Dataset, tmpdir: Path
):
    task = TaskType.CLASSIFICATION
    dataset = classification_dataset
    wrong_dataset = object_detection_dataset

    _check_invaild_dataset(task, dataset, wrong_dataset, tmpdir)
    _evaluate(task, dataset, tmpdir)


def test_od_evaluator(
    classification_dataset: Dataset, object_detection_dataset: Dataset, tmpdir: Path
):
    task = TaskType.OBJECT_DETECTION
    dataset = object_detection_dataset
    wrong_dataset = classification_dataset

    _check_invaild_dataset(task, dataset, wrong_dataset, tmpdir)
    _evaluate(task, dataset, tmpdir)


def test_ins_seg_evaluator(
    classification_dataset: Dataset, instance_segmentation_dataset: Dataset, tmpdir: Path
):
    task = TaskType.INSTANCE_SEGMENTATION
    dataset = instance_segmentation_dataset
    wrong_dataset = classification_dataset

    _check_invaild_dataset(task, dataset, wrong_dataset, tmpdir)
    _evaluate(task, dataset, tmpdir)


def test_sem_seg_evaluator(
    classification_dataset: Dataset, semantic_segmentation_dataset: Dataset, tmpdir: Path
):
    task = TaskType.SEMANTIC_SEGMENTATION
    dataset = semantic_segmentation_dataset
    wrong_dataset = classification_dataset

    _check_invaild_dataset(task, dataset, wrong_dataset, tmpdir)
    _evaluate(task, dataset, tmpdir)
