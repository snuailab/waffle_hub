from pathlib import Path
from typing import Any, Union

from torch.utils.data import DataLoader

from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.infer.callbacks import BaseInferenceCallback
from waffle_hub.hub.infer.inferencer import Inferencer


class InferenceStateWriterCallback(BaseInferenceCallback):
    def __init__(self, save_path: Union[str, Path]):
        self.save_path = Path(save_path)

    def before_inference(self, inferencer: Inferencer) -> None:
        """Called when the inference begins."""
        self._save_state(inferencer.state)

    def on_inference_loop_start(
        self, inferencer: Inferencer, dataset: Dataset, dataloader: DataLoader
    ) -> None:
        """Called when the inference loop begins."""
        self._save_state(inferencer.state)

    def on_inference_step_end(
        self, inferencer: Inferencer, step: int, batch: Any, result_batch: Any
    ) -> None:
        """Called when the inference loop step ends."""
        self._save_state(inferencer.state)

    def on_inference_end(self, inferencer: Inferencer) -> None:
        """Called when the inference function ends."""
        self._save_state(inferencer.state)

    def after_inference(self, inferencer: Inferencer) -> None:
        """Called when the inference ends."""
        self._save_state(inferencer.state)

    def on_exception_stopped(self, inferencer: Inferencer, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""
        self._save_state(inferencer.state)

    def on_exception_failed(self, inferencer: Inferencer, e: Exception) -> None:
        """Called when an error occurs"""
        self._save_state(inferencer.state)

    def _save_state(self, state):
        state.save_json(save_path=self.save_path)
