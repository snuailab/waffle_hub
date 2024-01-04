import torch
from waffle_utils.callback import BaseCallback
from waffle_utils.hook import BaseHook

from waffle_hub import TrainStatus
from waffle_hub.utils.metric_logger import MetricLogger
from waffle_hub.utils.process import _register_signal_handler


class BaseTrainHook(BaseHook):
    def __init__(self, callbacks: list[BaseCallback] = None):
        super().__init__(callbacks=callbacks)
        self.metric_logger: MetricLogger = None

    def setup(self) -> None:
        """Called when worker starts."""
        _register_signal_handler()
        self.metric_logger = MetricLogger(
            name=self.name,
            log_dir=self.train_log_dir,
            func=self.get_metrics,
            interval=10,
            prefix="waffle",
            state=self.state,
        )

    def teardown(self) -> None:
        """Called when worker ends."""
        if self.metric_logger is not None:
            self.metric_logger.stop()

    def before_train(self) -> None:
        """Called when the train begins."""
        self.state.total_step = self.cfg.epochs

    def on_train_start(self) -> None:
        """Called when the train function begins."""
        self.state.status = TrainStatus.RUNNING
        self.state.clear_error()
        self.metric_logger.start()

    def training(self) -> None:
        """Called when the training"""

    def on_train_end(self) -> None:
        """Called when the train function ends."""

    def after_train(self) -> None:
        """Called when the train ends."""
        # write result
        self.result.best_ckpt_file = self.best_ckpt_file
        self.result.last_ckpt_file = self.last_ckpt_file
        self.result.metrics = self.get_metrics()

        self.state.status = TrainStatus.SUCCESS

    def on_evaluate_start(self) -> None:
        """Called when the evaluate function begins."""

    def on_evaluate_end(self) -> None:
        """Called when the evaluate function ends."""

    def on_exception_stopped(self, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""
        self.state.status = TrainStatus.STOPPED
        self.state.set_error(e)

    def on_exception_failed(self, e: Exception) -> None:
        """Called when an error occurs"""
        self.state.status = TrainStatus.FAILED
        self.state.set_error(e)
