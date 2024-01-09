from collections import defaultdict
from pathlib import Path
from typing import Union

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from waffle_utils.file import io


# TODO: epochs logging & stop handling
class CustomCallback(TrainerCallback):
    """
    This class is necessary to obtain logs for the training.
    """

    def __init__(self, trainer, metric_file: Union[Path, str]) -> None:
        super().__init__()
        self._trainer = trainer
        self.metric_file = metric_file

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        epoch_metric = defaultdict(list)
        for metric in state.log_history:
            epoch = int(metric.get("epoch"))
            for key, value in metric.items():
                epoch_metric[epoch].append({"tag": key, "value": value})
        io.save_json(list(epoch_metric.values()), self.metric_file)
        return control
