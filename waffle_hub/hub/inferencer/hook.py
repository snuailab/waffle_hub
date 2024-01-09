from typing import Any

from torch.utils.data import DataLoader
from waffle_utils.callback import BaseCallback
from waffle_utils.hook import BaseHook

from waffle_hub import InferenceStatus
from waffle_hub.dataset.dataset import Dataset
from waffle_hub.utils.process import _register_signal_handler


class BaseInferenceHook(BaseHook):
    def __init__(self, callbacks: list[BaseCallback] = None):
        super().__init__(callbacks)

    def setup(self) -> None:
        """Called when worker starts."""
        _register_signal_handler()

    def teardown(self) -> None:
        """Called when worker ends."""

    def before_inference(self) -> None:
        """Called when the inference begins."""

    def on_inference_start(self) -> None:
        """Called when the inference function begins."""
        self.state.status = InferenceStatus.RUNNING
        self.state.clear_error()

    def on_inference_loop_start(self, dataset: Dataset, dataloader: DataLoader) -> None:
        """Called when the inference loop begins."""
        self.state.total_step = len(dataloader) + 1
        self.state.step = 0

    def on_inference_step_start(self, step: int, batch: Any) -> None:
        """Called when the inference loop step begins."""

    def on_inference_step_end(self, step: int, batch: Any, result_batch: Any) -> None:
        """Called when the inference loop step ends."""
        self.state.step = step

    def on_inference_loop_end(self, result: list[dict]) -> None:
        """Called when the inference loop ends."""

    def on_inference_end(self) -> None:
        """Called when the inference function ends."""

    def after_inference(self) -> None:
        """Called when the inference ends."""
        self.state.step = self.state.total_step
        self.state.status = InferenceStatus.SUCCESS

    def on_exception_stopped(self, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""
        self.state.status = InferenceStatus.STOPPED
        self.state.set_error(e)

    def on_exception_failed(self, e: Exception) -> None:
        """Called when an error occurs"""
        self.state.status = InferenceStatus.FAILED
        self.state.set_error(e)
