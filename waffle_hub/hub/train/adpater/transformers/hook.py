from typing import Any

from torch.utils.data import DataLoader
from waffle_utils.callback import BaseCallback
from waffle_utils.hook import BaseHook

from waffle_hub import EvaluateStatus


class BaseTrainHook(BaseHook):
    def __init__(self, callbacks: list[BaseCallback] = None):
        super().__init__(callbacks)

    def setup(self) -> None:
        """Called when worker starts."""

    def teardown(self) -> None:
        """Called when worker ends."""

    def before_train(self) -> None:
        """Called when the train begins."""

    def on_train_start(self) -> None:
        """Called when the train function begins."""

    def training(self) -> None:
        """Called when the training"""

    def on_train_end(self) -> None:
        """Called when the train function ends."""

    def after_train(self) -> None:
        """Called when the train ends."""

    def on_exception_stopped(self, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""

    def on_exception_failed(self, e: Exception) -> None:
        """Called when an error occurs"""
