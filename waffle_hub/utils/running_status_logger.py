from pathlib import Path

from waffle_hub import EvaluateStatus, ExportStatus, InferenceStatus, TrainStatus
from waffle_hub.schema.running_status import (
    STATUS_TYPE,
    BaseRunningStatus,
    EvaluatingStatus,
    ExportingStatus,
    InferencingStatus,
    TrainingStatus,
)


class RunningStatusLogger:
    running_status: BaseRunningStatus = None

    def __init__(self, running_status: BaseRunningStatus, save_path: Path):
        self.running_status = running_status
        if self.running_status is None:
            raise NotImplementedError("running_status must be not None")
        self.save_path = save_path

    def save(self):
        self.running_status.save_json(save_path=self.save_path)

    def set_status(self, status: STATUS_TYPE):
        self.running_status.status_desc = status
        self.save()

    def set_error(self, e: Exception = None):
        self.running_status.error_type = e.__class__.__name__
        self.running_status.error_msg = e
        self.save()

    def clear_error(self):
        self.running_status.error_type = None
        self.running_status.error_msg = None
        self.save()

    def clear_step(self):
        self.running_status.step = 0
        self.running_status.total_step = 0
        self.save()

    def set_total_step(self, total_step: int):
        self.running_status.total_step = total_step
        self.running_status.step = 0
        self.save()

    def set_current_step(self, step: int):
        self.running_status.step = step
        self.save()


class TrainingStatusLogger(RunningStatusLogger):
    def __init__(self, save_path: Path):
        super().__init__(TrainingStatus(), save_path)
        self.set_init()

    def set_init(self):
        self.clear_error()
        self.clear_step()
        self.set_status(TrainStatus.INIT)

    def set_failed(self, e):
        self.set_error(e)
        self.set_status(TrainStatus.FAILED)

    def set_success(self):
        self.clear_error()
        self.running_status.step = self.running_status.total_step
        self.set_status(TrainStatus.SUCCESS)

    def set_running(self):
        self.set_status(TrainStatus.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(TrainStatus.STOPPED)


class EvaluatingStatusLogger(RunningStatusLogger):
    def __init__(self, save_path: Path):
        super().__init__(EvaluatingStatus(), save_path)
        self.set_init()

    def set_init(self):
        self.clear_error()
        self.clear_step()
        self.set_status(EvaluateStatus.INIT)

    def set_failed(self, e):
        self.set_error(e)
        self.set_status(EvaluateStatus.FAILED)

    def set_success(self):
        self.clear_error()
        self.running_status.step = self.running_status.total_step
        self.set_status(EvaluateStatus.SUCCESS)

    def set_running(self):
        self.set_status(EvaluateStatus.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(EvaluateStatus.STOPPED)


class InferencingStatusLogger(RunningStatusLogger):
    def __init__(self, save_path: Path):
        super().__init__(InferencingStatus(), save_path)
        self.set_init()

    def set_init(self):
        self.clear_error()
        self.clear_step()
        self.set_status(InferenceStatus.INIT)

    def set_failed(self, e):
        self.set_error(e)
        self.set_status(InferenceStatus.FAILED)

    def set_success(self):
        self.clear_error()
        self.running_status.step = self.running_status.total_step
        self.set_status(InferenceStatus.SUCCESS)

    def set_running(self):
        self.set_status(InferenceStatus.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(InferenceStatus.STOPPED)


class ExportingStatusLogger(RunningStatusLogger):
    def __init__(self, save_path: Path):
        super().__init__(ExportingStatus(), save_path)
        self.set_init()

    def set_init(self):
        self.clear_error()
        self.clear_step()
        self.set_status(ExportStatus.INIT)

    def set_failed(self, e):
        self.set_error(e)
        self.set_status(ExportStatus.FAILED)

    def set_success(self):
        self.clear_error()
        self.running_status.step = self.running_status.total_step
        self.set_status(ExportStatus.SUCCESS)

    def set_running(self):
        self.set_status(ExportStatus.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(ExportStatus.STOPPED)
