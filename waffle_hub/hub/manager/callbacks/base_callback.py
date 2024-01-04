from waffle_utils.callback import BaseCallback

from waffle_hub.hub.manager.base_manager import BaseManager


class BaseTrainCallback(BaseCallback):
    def setup(self, manager: BaseManager) -> None:
        """Called when worker starts."""

    def teardown(self, manager: BaseManager) -> None:
        """Called when worker ends."""

    def before_train(self, manager: BaseManager) -> None:
        """Called when the train begins."""

    def on_train_start(self, manager: BaseManager) -> None:
        """Called when the train function begins."""

    def training(self, manager: BaseManager) -> None:
        """Called when the training"""

    def on_train_end(self, manager: BaseManager) -> None:
        """Called when the train function ends."""

    def after_train(self, manager: BaseManager) -> None:
        """Called when the train ends."""

    def on_evaluate_start(self, manager: BaseManager) -> None:
        """Called when the evaluate function begins."""

    def on_evaluate_end(self, manager: BaseManager) -> None:
        """Called when the evaluate function ends."""

    def on_exception_stopped(self, manager: BaseManager, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""

    def on_exception_failed(self, manager: BaseManager, e: Exception) -> None:
        """Called when an error occurs"""
