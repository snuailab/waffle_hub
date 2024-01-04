from ctypes import Union
from pathlib import Path

from waffle_hub.hub.train.callbacks import BaseTrainCallback
from waffle_hub.hub.train.trainer import Trainer


class TrainStateWriterCallback(BaseTrainCallback):
    def __init__(self, save_path: Union[str, Path]):
        self.save_path = Path(save_path)

    def setup(self, trainer: Trainer) -> None:
        """Called when worker starts."""
        self._save_state(trainer.state)
        if hasattr(trainer, "metric_logger"):
            trainer.metric_logger.set_state_save_path(self.save_path)

    def teardown(self, trainer: Trainer) -> None:
        """Called when worker ends."""
        self._save_state(trainer.state)

    def before_train(self, trainer: Trainer) -> None:
        """Called when the train begins."""
        self._save_state(trainer.state)

    def on_train_start(self, trainer: Trainer) -> None:
        """Called when the train function begins."""
        self._save_state(trainer.state)

    def training(self, trainer: Trainer) -> None:
        """Called when the training"""

    def on_train_end(self, trainer: Trainer) -> None:
        """Called when the train function ends."""

    def after_train(self, trainer: Trainer) -> None:
        """Called when the train ends."""
        self._save_state(trainer.state)

    def on_exception_stopped(self, trainer: Trainer, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""
        self._save_state(trainer.state)

    def on_exception_failed(self, trainer: Trainer, e: Exception) -> None:
        """Called when an error occurs"""
        self._save_state(trainer.state)

    def _save_state(self, state):
        state.save_json(save_path=self.save_path)
