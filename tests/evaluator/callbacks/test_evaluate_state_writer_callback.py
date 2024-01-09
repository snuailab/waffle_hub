from pathlib import Path

from waffle_utils.file import io

from waffle_hub import EvaluateStatus
from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.evaluator.callbacks import EvaluateStateWriterCallback
from waffle_hub.hub.evaluator.evaluator import Evaluator
from waffle_hub.hub.model.boring_wrapper import BoringModelWrapper
from waffle_hub.type import TaskType


def test_evaluate_state_writer_callback(classification_dataset: Dataset, tmpdir: Path):
    # set up
    dataset = classification_dataset  # for raw image
    task = TaskType.CLASSIFICATION
    state_save_path = tmpdir / "evaluate_state.json"

    model = BoringModelWrapper(task=task, categories=["1", "2"], batch_size=1)
    evaluator = Evaluator(
        root_dir=tmpdir,
        model=model,
        callbacks=[EvaluateStateWriterCallback(save_path=state_save_path)],
    )

    assert evaluator.state.status == EvaluateStatus.INIT
    assert not state_save_path.exists()

    result = evaluator.evaluate(
        dataset=dataset,
        batch_size=1,
        image_size=[32, 32],
        letter_box=False,
        device="cpu",
    )

    assert evaluator.state.status == EvaluateStatus.SUCCESS
    assert state_save_path.exists()
    temp_state = io.load_json(state_save_path)
    assert temp_state["status"] == EvaluateStatus.SUCCESS
    assert temp_state["step"] == temp_state["total_step"]
