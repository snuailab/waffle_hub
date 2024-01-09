from pathlib import Path

from waffle_hub import InferenceStatus
from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.inferencer.callbacks import InferenceDrawCallback
from waffle_hub.hub.inferencer.inferencer import Inferencer
from waffle_hub.hub.model.boring_wrapper import BoringModelWrapper
from waffle_hub.type import TaskType


def test_register_inference_draw_callback(classification_dataset: Dataset, tmpdir: Path):
    # set up
    dataset = classification_dataset  # for raw image
    task = TaskType.CLASSIFICATION
    draw_dir = tmpdir / "draw"

    model = BoringModelWrapper(task=task, categories=["1", "2"], batch_size=1)
    inferencer = Inferencer(
        root_dir=tmpdir, model=model, callbacks=[InferenceDrawCallback(draw_dir=draw_dir)]
    )

    assert not draw_dir.exists()

    result = inferencer.inference(
        source=dataset.raw_image_dir,
        recursive=True,
        image_size=[32, 32],
        letter_box=False,
        batch_size=1,
        half=False,
        workers=0,
        device="cpu",
    )

    assert inferencer.state.status == InferenceStatus.SUCCESS
    assert draw_dir.exists()


def test_inference_draw_option(classification_dataset: Dataset, tmpdir: Path):
    # set up
    dataset = classification_dataset  # for raw image
    task = TaskType.CLASSIFICATION

    model = BoringModelWrapper(task=task, categories=["1", "2"], batch_size=1)
    inferencer = Inferencer(root_dir=tmpdir, model=model)

    assert not inferencer.draw_dir.exists()

    result = inferencer.inference(
        source=dataset.raw_image_dir,
        recursive=True,
        image_size=[32, 32],
        letter_box=False,
        batch_size=1,
        half=False,
        workers=0,
        draw=True,
        device="cpu",
    )

    assert inferencer.state.status == InferenceStatus.SUCCESS
    assert inferencer.draw_dir.exists()
