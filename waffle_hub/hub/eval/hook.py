from typing import Any

from torch.utils.data import DataLoader
from waffle_utils.callback import BaseCallback
from waffle_utils.hook import BaseHook

from waffle_hub import EvaluateStatus
from waffle_hub.utils.process import _register_signal_handler


class BaseEvaluateHook(BaseHook):
    def __init__(self, callbacks: list[BaseCallback] = None):
        super().__init__(callbacks)

    def setup(self) -> None:
        """Called when worker starts."""
        _register_signal_handler()

    def teardown(self) -> None:
        """Called when worker ends."""

    def before_evaluate(self) -> None:
        """Called when the evaluate begins."""

    def on_evaluate_start(self) -> None:
        """Called when the evaluate function begins."""
        self.state.status = EvaluateStatus.RUNNING
        self.state.clear_error()

    def on_evaluate_loop_start(self, dataloader: DataLoader) -> None:
        """Called when the evaluate loop begins."""
        self.state.total_step = len(dataloader) + 1
        self.state.step = 0

    def on_evaluate_step_start(self, step: int, batch: Any) -> None:
        """Called when the evaluate loop step begins."""

    def on_evaluate_step_end(self, step: int, batch: Any, result_batch: Any) -> None:
        """Called when the evaluate loop step ends."""
        self.state.step = step

    def on_evaluate_loop_end(self, preds: Any) -> None:
        """Called when the evaluate loop ends."""

    def on_evaluate_end(self, result_metrics: list[dict]) -> None:
        """Called when the evaluate function ends."""
        self.result.eval_metrics = result_metrics

    def after_evaluate(self) -> None:
        """Called when the evaluate ends."""
        self.state.step = self.state.total_step
        self.state.status = EvaluateStatus.SUCCESS

    def on_exception_stopped(self, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""
        self.state.status = EvaluateStatus.STOPPED
        self.state.set_error(e)

    def on_exception_failed(self, e: Exception) -> None:
        """Called when an error occurs"""
        self.state.status = EvaluateStatus.FAILED
        self.state.set_error(e)
