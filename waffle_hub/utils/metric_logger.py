import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Thread
from typing import Union

__all__ = ["MetricLogger"]

logger = logging.getLogger(__name__)


metric_logger_classes = []


class _BaseMetricLogger(ABC):
    def __init__(self, name, log_dir, **kwargs):
        self.name = str(name)
        self.log_dir = Path(log_dir)

    @abstractmethod
    def log_metric(self, tag, value, step):
        raise NotImplementedError

    @abstractmethod
    def open(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


try:
    from torch.utils.tensorboard import SummaryWriter

    class _TensorboardLogger(_BaseMetricLogger):
        def __init__(self, name, log_dir, **kwargs):
            super().__init__(name, log_dir, **kwargs)

            self.writer = None

        def log_metric(self, tag, value, step):
            if self.writer is None:
                raise RuntimeError("Tensorboard logger is not opened.")
            self.writer.add_scalar(tag, value, step)

        def open(self):
            self.writer = SummaryWriter(self.log_dir)
            logging.info(f"Tensorboard logs will be saved to {self.log_dir}")

        def close(self):
            self.writer.close()
            self.writer = None

    metric_logger_classes.append(_TensorboardLogger)

except:
    logger.warning(
        "Tensorboard is not installed. To use Tensorboard logger, please install it with `pip install tensorboard`"
    )

# try:
#     import mlflow

#     class _MLFlowLogger(_BaseMetricLogger):
#         def __init__(self, name, log_dir, **kwargs):
#             super().__init__(name, log_dir, **kwargs)

#             self.kwargs = kwargs

#         def log_metric(self, tag, value, step):
#             # remove invalid characters in tag using regex (only alphabet, -, _, / are allowed)
#             tag = re.sub(r"[^a-zA-Z0-9-_\/]", "_", tag)
#             with mlflow.start_run(run_name=self.kwargs.get("run_name", "0")):
#                 mlflow.log_metric(tag, value, step)

#         def open(self):
#             exp = mlflow.set_experiment(self.name)
#             # TODO: only support one model for one experiment
#             for run in mlflow.search_runs(experiment_ids=exp.experiment_id).run_id:
#                 mlflow.delete_run(mlflow.get_run(run).info.run_id)
#             logging.info(f"MLFlow logs will be saved to {mlflow.get_tracking_uri()}")

#         def close(self):
#             mlflow.end_run()

#     metric_logger_classes.append(_MLFlowLogger)

# except:
#     logger.warning(
#         "MLFlow is not installed. To use MLFlow logger, please install it with `pip install mlflow`"
#     )


class MetricLogger:
    STEP_KEYWORDS = ["step", "epoch", "iter", "iteration"]
    MAX_TRY = 10

    def __init__(
        self,
        name: str,
        log_dir: Union[str, Path],
        func: callable,
        interval: float,
        prefix: str = "",
        **kwargs,
    ):
        """A metric logger that logs metrics every interval seconds.

        Args:
            name (str): The name of the metric logger.
            log_dir (Union[str, Path]): The directory to save the logs.
            func (Callable): The function to get the metrics.
                func should return a list of metrics.
                e.g. func() -> [[{"tag": "value}, ...], ...]
            interval (float): The interval to log metrics. (seconds)
            prefix (str, optional): The prefix of the log file. Defaults to "".
            kwargs: The arguments for the metric logger.
        """

        self.name = str(name)
        self.log_dir = log_dir
        self.func = func
        self.interval = float(interval)
        self.prefix = str(prefix)

        self.kwargs = kwargs

        self.loggers = [
            logger(self.name, self.log_dir, **kwargs) for logger in metric_logger_classes
        ]

        self._stop = False
        self._thread = None
        self._initiated = False
        self._last_step = 0

    def log_metric(self, tag, value, step):
        for logger in self.loggers:
            logger.log_metric((self.prefix + "/" if self.prefix else "") + tag, value, step)

    def start(self):
        """Start logging thread."""
        if self._initiated:
            raise RuntimeError("Metric logger has already been initiated.")

        self._initiated = True

        for logger in self.loggers:
            logger.open()

        self._stop = False

        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop logging thread."""
        self._stop = True

        self.thread.join()
        for logger in self.loggers:
            logger.close()

    def _loop(self):
        """Log metrics every interval seconds."""
        while not self._stop:
            chance = self.MAX_TRY
            while True:
                try:
                    self._log()
                    break
                except Exception as e:
                    chance -= 1
                    if chance == 0:
                        raise e
                    time.sleep(1)
            time.sleep(self.interval)
        self._log()  # final log after stop

    def _log(self):
        """Log metrics."""
        metrics_per_epoch = self.func()
        current_step = len(metrics_per_epoch)
        for step in range(self._last_step, current_step):
            # metrics is a list of dict
            # e.g. [{"tag": "value"}, ...]
            metric_dict = {v["tag"]: v["value"] for v in metrics_per_epoch[step]}

            # pop all keys from metrics if they contains step keywords
            for key in list(metric_dict.keys()):
                for step_keyword in self.STEP_KEYWORDS:
                    if step_keyword == key.lower():
                        metric_dict.pop(key)

            for tag, value in metric_dict.items():
                self.log_metric(tag, value, step)

        self._last_step = current_step
