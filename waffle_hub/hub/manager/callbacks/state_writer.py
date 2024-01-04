from pathlib import Path
from typing import Union

from waffle_hub.hub.evaluator.callbacks import EvaluateStateWriterCallback
from waffle_hub.hub.manager.base_manager import BaseManager
from waffle_hub.hub.manager.callbacks import BaseTrainCallback


class TrainStateWriterCallback(BaseTrainCallback):
    def __init__(
        self, train_state_save_path: Union[str, Path], eval_state_save_path: Union[str, Path]
    ):
        self.train_state_save_path = Path(train_state_save_path)
        self.eval_state_save_path = Path(eval_state_save_path)

    def setup(self, manager: BaseManager) -> None:
        """Called when worker starts."""
        self._save_state(manager.state)
        if hasattr(manager, "metric_logger"):
            manager.metric_logger.set_state_save_path(self.train_state_save_path)

    def teardown(self, manager: BaseManager) -> None:
        """Called when worker ends."""
        self._save_state(manager.state)

    def before_train(self, manager: BaseManager) -> None:
        """Called when the train begins."""
        self._save_state(manager.state)

    def on_train_start(self, manager: BaseManager) -> None:
        """Called when the train function begins."""
        self._save_state(manager.state)

    def training(self, manager: BaseManager) -> None:
        """Called when the training"""

    def on_train_end(self, manager: BaseManager) -> None:
        """Called when the train function ends."""

    def after_train(self, manager: BaseManager) -> None:
        """Called when the train ends."""
        self._save_state(manager.state)

    def on_evaluate_start(self, manager: BaseManager) -> None:
        """Called when the evaluate function begins."""
        manager.evaluator.register_callback(
            EvaluateStateWriterCallback(save_path=self.eval_state_save_path)
        )

    def on_evaluate_end(self, manager: BaseManager) -> None:
        """Called when the evaluate function ends."""

    def on_exception_stopped(self, manager: BaseManager, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""
        self._save_state(manager.state)

    def on_exception_failed(self, manager: BaseManager, e: Exception) -> None:
        """Called when an error occurs"""
        self._save_state(manager.state)

    def _save_state(self, state):
        state.save_json(save_path=self.train_state_save_path)
