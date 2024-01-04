from pathlib import Path
from typing import Any, Union

from torch.utils.data import DataLoader

from waffle_hub.hub.eval.callbacks import BaseEvaluateCallback
from waffle_hub.hub.eval.evaluator import Evaluator


class EvaluateStateWriterCallback(BaseEvaluateCallback):
    def __init__(self, save_path: Union[str, Path]):
        self.save_path = Path(save_path)

    def before_evaluate(self, evaluator: Evaluator) -> None:
        self._save_state(evaluator.state)

    def on_evaluate_start(self, evaluator: Evaluator) -> None:
        """Called when the evaluate function begins."""
        self._save_state(evaluator.state)

    def on_evaluate_loop_start(self, evaluator: Evaluator, dataloader: DataLoader) -> None:
        """Called when the evaluate loop begins."""
        self._save_state(evaluator.state)

    def on_evaluate_step_end(
        self, evaluator: Evaluator, step: int, batch: Any, result_batch: Any
    ) -> None:
        """Called when the evaluate loop step ends."""
        self._save_state(evaluator.state)

    def after_evaluate(self, evaluator: Evaluator) -> None:
        """Called when the evaluate ends."""
        self._save_state(evaluator.state)

    def on_exception_stopped(self, evaluator: Evaluator, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""
        self._save_state(evaluator.state)

    def on_exception_failed(self, evaluator: Evaluator, e: Exception) -> None:
        """Called when an error occurs"""
        self._save_state(evaluator.state)

    def _save_state(self, state):
        state.save_json(save_path=self.save_path)
