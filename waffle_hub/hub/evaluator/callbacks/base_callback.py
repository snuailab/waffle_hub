from abc import ABC
from typing import Any

from torch.utils.data import DataLoader
from waffle_utils.callback import BaseCallback

from waffle_hub.hub.evaluator.evaluator import Evaluator


class BaseEvaluateCallback(BaseCallback, ABC):
    def __init__(self):
        pass

    def setup(self, evaluator: Evaluator) -> None:
        """Called when worker starts."""

    def teardown(self, evaluator: Evaluator) -> None:
        """Called when worker ends."""

    def before_evaluate(self, evaluator: Evaluator) -> None:
        """Called when the evaluate begins."""

    def on_evaluate_start(self, evaluator: Evaluator) -> None:
        """Called when the evaluate function begins."""

    def on_evaluate_loop_start(self, evaluator: Evaluator, dataloader: DataLoader) -> None:
        """Called when the evaluate loop begins."""

    def on_evaluate_step_start(self, evaluator: Evaluator, step: int, batch: Any) -> None:
        """Called when the evaluate loop step begins."""

    def on_evaluate_step_end(
        self, evaluator: Evaluator, step: int, batch: Any, result_batch: Any
    ) -> None:
        """Called when the evaluate loop step ends."""

    def on_evaluate_loop_end(self, evaluator: Evaluator, preds: Any) -> None:
        """Called when the evaluate loop ends."""

    def on_evaluate_end(self, evaluator: Evaluator, result_metrics: list[dict]) -> None:
        """Called when the evaluate function ends."""

    def after_evaluate(self, evaluator: Evaluator) -> None:
        """Called when the evaluate ends."""

    def on_exception_stopped(self, evaluator: Evaluator, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""

    def on_exception_failed(self, evaluator: Evaluator, e: Exception) -> None:
        """Called when an error occurs"""
