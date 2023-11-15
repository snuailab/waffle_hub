from pathlib import Path

from waffle_hub import (
    EvaluateStatusDesc,
    ExportOnnxStatusDesc,
    ExportWaffleStatusDesc,
    InferenceStatusDesc,
    TrainStatusDesc,
)
from waffle_hub.schema.running_status import (
    STATUS_DESC_TYPE,
    BaseRunningStatus,
    EvaluatingStatus,
    ExportingOnnxStatus,
    ExportingWaffleStatus,
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

    def set_status(self, status: STATUS_DESC_TYPE):
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
        self.set_status(TrainStatusDesc.INIT)

    def set_failed(self, e):
        self.set_error(e)
        self.set_status(TrainStatusDesc.FAILED)

    def set_success(self):
        self.clear_error()
        self.running_status.step = self.running_status.total_step
        self.set_status(TrainStatusDesc.SUCCESS)

    def set_running(self):
        self.set_status(TrainStatusDesc.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(TrainStatusDesc.STOPPED)


class EvaluatingStatusLogger(RunningStatusLogger):
    def __init__(self, save_path: Path):
        super().__init__(EvaluatingStatus(), save_path)
        self.set_init()

    def set_init(self):
        self.clear_error()
        self.clear_step()
        self.set_status(EvaluateStatusDesc.INIT)

    def set_failed(self, e):
        self.set_error(e)
        self.set_status(EvaluateStatusDesc.FAILED)

    def set_success(self):
        self.clear_error()
        self.running_status.step = self.running_status.total_step
        self.set_status(EvaluateStatusDesc.SUCCESS)

    def set_running(self):
        self.set_status(EvaluateStatusDesc.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(EvaluateStatusDesc.STOPPED)


class InferencingStatusLogger(RunningStatusLogger):
    def __init__(self, save_path: Path):
        super().__init__(InferencingStatus(), save_path)
        self.set_init()

    def set_init(self):
        self.clear_error()
        self.clear_step()
        self.set_status(InferenceStatusDesc.INIT)

    def set_failed(self, e):
        self.set_error(e)
        self.set_status(InferenceStatusDesc.FAILED)

    def set_success(self):
        self.clear_error()
        self.running_status.step = self.running_status.total_step
        self.set_status(InferenceStatusDesc.SUCCESS)

    def set_running(self):
        self.set_status(InferenceStatusDesc.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(InferenceStatusDesc.STOPPED)


class ExportingOnnxStatusLogger(RunningStatusLogger):
    def __init__(self, save_path: Path):
        super().__init__(ExportingOnnxStatus(), save_path)
        self.set_init()

    def set_init(self):
        self.clear_error()
        self.clear_step()
        self.set_status(ExportOnnxStatusDesc.INIT)

    def set_failed(self, e):
        self.set_error(e)
        self.set_status(ExportOnnxStatusDesc.FAILED)

    def set_success(self):
        self.clear_error()
        self.running_status.step = self.running_status.total_step
        self.set_status(ExportOnnxStatusDesc.SUCCESS)

    def set_running(self):
        self.set_status(ExportOnnxStatusDesc.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(ExportOnnxStatusDesc.STOPPED)


class ExportingWaffleStatusLogger(RunningStatusLogger):
    def __init__(self, save_path: Path):
        super().__init__(ExportingWaffleStatus(), save_path)
        self.set_init()

    def set_init(self):
        self.clear_error()
        self.clear_step()
        self.set_status(ExportWaffleStatusDesc.INIT)

    def set_failed(self, e):
        self.set_error(e)
        self.set_status(ExportWaffleStatusDesc.FAILED)

    def set_success(self):
        self.clear_error()
        self.running_status.step = self.running_status.total_step
        self.set_status(ExportWaffleStatusDesc.SUCCESS)

    def set_running(self):
        self.set_status(ExportWaffleStatusDesc.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(ExportWaffleStatusDesc.STOPPED)
