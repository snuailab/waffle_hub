from pathlib import Path

from waffle_hub import EvaluateStatus, ExportStatus, InferenceStatus, TrainStatus
from waffle_hub.schema.working_info import (
    STATUS_TYPE,
    BaseWorkingInfo,
    EvaluatingInfo,
    ExportingInfo,
    InferencingInfo,
    TrainingInfo,
)


class RunningInfoLogger:
    working_info: BaseWorkingInfo = None

    def __init__(self, working_info: BaseWorkingInfo, save_path: Path):
        self.working_info = working_info
        if self.working_info is None:
            raise NotImplementedError("working_info must be not None")
        self.save_path = save_path

    def save(self):
        self.working_info.save_json(save_path=self.save_path)

    def set_status(self, status: STATUS_TYPE):
        self.working_info.status = status
        self.save()

    def set_error(self, e: Exception = None):
        self.working_info.error_type = e.__class__.__name__
        self.working_info.error_msg = e
        self.save()

    def clear_error(self):
        self.working_info.error_type = None
        self.working_info.error_msg = None
        self.save()

    def clear_step(self):
        self.working_info.step = 0
        self.working_info.total_step = 0
        self.save()

    def set_total_step(self, total_step: int):
        self.working_info.total_step = total_step
        self.working_info.step = 0
        self.save()

    def set_current_step(self, step: int):
        self.working_info.step = step
        self.save()


class TrainingInfoLogger(RunningInfoLogger):
    def __init__(self, save_path: Path):
        super().__init__(TrainingInfo(), save_path)
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
        self.working_info.step = self.working_info.total_step
        self.set_status(TrainStatus.SUCCESS)

    def set_running(self):
        self.set_status(TrainStatus.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(TrainStatus.STOPPED)


class EvaluatingInfoLogger(RunningInfoLogger):
    def __init__(self, save_path: Path):
        super().__init__(EvaluatingInfo(), save_path)
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
        self.working_info.step = self.working_info.total_step
        self.set_status(EvaluateStatus.SUCCESS)

    def set_running(self):
        self.set_status(EvaluateStatus.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(EvaluateStatus.STOPPED)


class InferencingInfoLogger(RunningInfoLogger):
    def __init__(self, save_path: Path):
        super().__init__(InferencingInfo(), save_path)
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
        self.working_info.step = self.working_info.total_step
        self.set_status(InferenceStatus.SUCCESS)

    def set_running(self):
        self.set_status(InferenceStatus.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(InferenceStatus.STOPPED)


class ExportingInfoLogger(RunningInfoLogger):
    def __init__(self, save_path: Path):
        super().__init__(ExportingInfo(), save_path)
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
        self.working_info.step = self.working_info.total_step
        self.set_status(ExportStatus.SUCCESS)

    def set_running(self):
        self.set_status(ExportStatus.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(ExportStatus.STOPPED)
