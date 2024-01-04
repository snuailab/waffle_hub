from waffle_utils.callback import BaseCallback

from waffle_hub.hub.train.base_trainer import Trainer


class BaseTrainCallback(BaseCallback):
    def setup(self, trainer: Trainer) -> None:
        """Called when worker starts."""

    def teardown(self, trainer: Trainer) -> None:
        """Called when worker ends."""

    def before_train(self, trainer: Trainer) -> None:
        """Called when the train begins."""

    def on_train_start(self, trainer: Trainer) -> None:
        """Called when the train function begins."""

    def training(self, trainer: Trainer) -> None:
        """Called when the training"""

    def on_train_end(self, trainer: Trainer) -> None:
        """Called when the train function ends."""

    def after_train(self, trainer: Trainer) -> None:
        """Called when the train ends."""

    def on_exception_stopped(self, trainer: Trainer, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""

    def on_exception_failed(self, trainer: Trainer, e: Exception) -> None:
        """Called when an error occurs"""
