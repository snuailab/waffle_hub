from pathlib import Path

from waffle_utils.file import io

from waffle_hub import EvaluateStatus, TrainStatus
from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.manager.adapter.ultralytics.ultralytics import UltralyticsManager
from waffle_hub.hub.manager.callbacks import TrainStateWriterCallback
from waffle_hub.type import TaskType


def test_train_state_writer_callback(classification_dataset: Dataset, tmpdir: Path):
    # set up
    dataset = classification_dataset  # for raw image
    task = TaskType.CLASSIFICATION
    model_type = "yolov8"
    model_size = "n"

    train_state_save_path = tmpdir / "train_state.json"
    evaluate_state_save_path = tmpdir / "evaluate_state.json"

    manager = UltralyticsManager(
        root_dir=tmpdir,
        name="test",
        task=task,
        model_type=model_type,
        model_size=model_size,
        callbacks=[
            TrainStateWriterCallback(
                train_state_save_path=train_state_save_path,
                eval_state_save_path=evaluate_state_save_path,
            )
        ],
    )
    assert manager.state.status == TrainStatus.INIT
    assert not train_state_save_path.exists()
    assert not evaluate_state_save_path.exists()

    result = manager.train(
        dataset=dataset,
        epochs=1,
        batch_size=4,
        image_size=[32, 32],
        device="cpu",
        workers=0,
    )

    assert manager.state.status == TrainStatus.SUCCESS
    assert train_state_save_path.exists()
    assert evaluate_state_save_path.exists()

    temp_train_state = io.load_json(train_state_save_path)
    assert temp_train_state["status"] == TrainStatus.SUCCESS
    assert temp_train_state["step"] == temp_train_state["total_step"]

    temp_evaluate_state = io.load_json(evaluate_state_save_path)
    assert temp_evaluate_state["status"] == EvaluateStatus.SUCCESS
    assert temp_evaluate_state["step"] == temp_evaluate_state["total_step"]
