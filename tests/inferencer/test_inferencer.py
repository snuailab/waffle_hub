from pathlib import Path

import pytest
from waffle_utils.file import io

from waffle_hub import InferenceStatus
from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.inferencer.inferencer import Inferencer
from waffle_hub.hub.model.boring_wrapper import BoringModelWrapper
from waffle_hub.type import TaskType


@pytest.mark.parametrize(
    "task",
    [
        TaskType.CLASSIFICATION,
        TaskType.OBJECT_DETECTION,
        TaskType.SEMANTIC_SEGMENTATION,
        TaskType.INSTANCE_SEGMENTATION,
    ],
)
def test_inferencer(classification_dataset: Dataset, tmpdir: Path, task: TaskType):
    dataset = classification_dataset  # for raw image
    model = BoringModelWrapper(task=task, categories=["1", "2", "3"], batch_size=1)

    inferencer = Inferencer(root_dir=tmpdir, model=model)

    assert inferencer.state.status == InferenceStatus.INIT

    result = inferencer.inference(
        source=dataset.raw_image_dir,
        recursive=True,
        image_size=[32, 32],
        letter_box=False,
        batch_size=1,
        confidence_threshold=0.5,
        iou_threshold=0.5,
        half=False,
        workers=0,
        device="cpu",
    )

    assert inferencer.state.status == InferenceStatus.SUCCESS
    assert inferencer.result == result
    assert inferencer.inference_file.exists()
    assert io.load_json(inferencer.inference_file) == result.predictions
