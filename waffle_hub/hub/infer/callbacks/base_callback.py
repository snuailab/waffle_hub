from abc import ABC
from typing import Any

from torch.utils.data import DataLoader
from waffle_utils.callback import BaseCallback

from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.infer.inferencer import Inferencer


class BaseInferenceCallback(BaseCallback, ABC):
    def __init__(self):
        pass

    def setup(self, inferencer: Inferencer) -> None:
        """Called when worker starts."""

    def teardown(self, inferencer: Inferencer) -> None:
        """Called when worker ends."""

    def before_inference(self, inferencer: Inferencer) -> None:
        """Called when the inference begins."""

    def on_inference_start(self, inferencer: Inferencer) -> None:
        """Called when the inference function begins."""

    def on_inference_loop_start(
        self, inferencer: Inferencer, dataset: Dataset, dataloader: DataLoader
    ) -> None:
        """Called when the inference loop begins."""

    def on_inference_step_start(self, inferencer: Inferencer, step: int, batch: Any) -> None:
        """Called when the inference loop step begins."""

    def on_inference_step_end(
        self, inferencer: Inferencer, step: int, batch: Any, result_batch: Any
    ) -> None:
        """Called when the inference loop step ends."""

    def on_inference_loop_end(self, inferencer: Inferencer, result: list[dict]) -> None:
        """Called when the inference loop ends."""

    def on_inference_end(self, inferencer: Inferencer) -> None:
        """Called when the inference function ends."""

    def after_inference(self, inferencer: Inferencer) -> None:
        """Called when the inference ends."""

    def on_exception_stopped(self, inferencer: Inferencer, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""

    def on_exception_failed(self, inferencer: Inferencer, e: Exception) -> None:
        """Called when an error occurs"""
