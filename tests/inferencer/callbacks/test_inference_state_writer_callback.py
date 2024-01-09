from pathlib import Path

from waffle_utils.file import io

from waffle_hub import InferenceStatus
from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.inferencer.callbacks import InferenceStateWriterCallback
from waffle_hub.hub.inferencer.inferencer import Inferencer
from waffle_hub.hub.model.boring_wrapper import BoringModelWrapper
from waffle_hub.type import TaskType


def test_inference_state_writer_callback(classification_dataset: Dataset, tmpdir: Path):
    # set up
    dataset = classification_dataset  # for raw image
    task = TaskType.CLASSIFICATION
    state_save_path = tmpdir / "inference_state.json"

    model = BoringModelWrapper(task=task, categories=["1", "2"], batch_size=1)
    inferencer = Inferencer(
        root_dir=tmpdir,
        model=model,
        callbacks=[InferenceStateWriterCallback(save_path=state_save_path)],
    )

    assert inferencer.state.status == InferenceStatus.INIT
    assert not state_save_path.exists()

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
    assert state_save_path.exists()

    temp_state = io.load_json(state_save_path)
    assert temp_state["status"] == InferenceStatus.SUCCESS
    assert temp_state["step"] == temp_state["total_step"]
