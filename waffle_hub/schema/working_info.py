from dataclasses import dataclass

from waffle_hub import (
    BaseEnum,
    EvaluateStatus,
    ExportStatus,
    InferenceStatus,
    TrainStatus,
)
from waffle_hub.schema.base_schema import BaseSchema


@dataclass
class BaseWorkingInfo(BaseSchema):
    error_type: str = None
    error_msg: str = None
    step: int = None
    total_step: int = None


@dataclass
class TrainingInfo(BaseWorkingInfo):
    status: TrainStatus = None


@dataclass
class EvaluatingInfo(BaseWorkingInfo):
    status: EvaluateStatus = None


@dataclass
class InferencingInfo(BaseWorkingInfo):
    status: InferenceStatus = None


@dataclass
class ExportingInfo(BaseWorkingInfo):
    status: ExportStatus = None


class BaseInfoController:
    status_enum: BaseEnum = None
    working_info: BaseWorkingInfo = None

    def __init__(self, save_path):
        self.save_path = save_path

    def save(self):
        self.working_info.save_json(save_path=self.save_path)

    def set_status(self, status, e=None):
        if not status in self.status_enum:
            raise ValueError(f"status must be one of {list(self.status_enum)}")
        self.working_info.status = status
        if e:
            self.working_info.error_type = e.__class__.__name__
            self.working_info.error_msg = e
        else:
            self.working_info.error_type = None
            self.working_info.error_msg = None
        self.save()

    def set_total_step(self, total_step):
        self.working_info.total_step = total_step
        self.working_info.step = 0
        self.save()

    def set_current_step(self, step):
        self.working_info.step = step
        self.save()


class TrainingInfoController(BaseInfoController):
    def __init__(self, save_path):
        super().__init__(save_path)
        self.working_info = TrainingInfo()
        self.status_enum = TrainStatus
        self.set_init()

    def set_init(self):
        self.set_status(self.status_enum.INIT)

    def set_failed(self, e):
        self.set_status(self.status_enum.FAILED, e)

    def set_success(self):
        self.working_info.step = self.working_info.total_step
        self.set_status(self.status_enum.SUCCESS)

    def set_running(self):
        self.set_status(self.status_enum.RUNNING)

    def set_stopped(self, e):
        self.set_status(self.status_enum.STOPPED, e)


class EvaluatingInfoController(BaseInfoController):
    def __init__(self, save_path):
        super().__init__(save_path)
        self.working_info = EvaluatingInfo()
        self.status_enum = EvaluateStatus
        self.set_init()

    def set_init(self):
        self.set_status(self.status_enum.INIT)

    def set_failed(self, e):
        self.set_status(self.status_enum.FAILED, e)

    def set_success(self):
        self.working_info.step = self.working_info.total_step
        self.set_status(self.status_enum.SUCCESS)

    def set_running(self):
        self.set_status(self.status_enum.RUNNING)

    def set_stopped(self, e):
        self.set_status(self.status_enum.STOPPED, e)


class InferencingInfoController(BaseInfoController):
    def __init__(self, save_path):
        super().__init__(save_path)
        self.working_info = InferencingInfo()
        self.status_enum = InferenceStatus
        self.set_init()

    def set_init(self):
        self.set_status(self.status_enum.INIT)

    def set_failed(self, e):
        self.set_status(self.status_enum.FAILED, e)

    def set_success(self):
        self.working_info.step = self.working_info.total_step
        self.set_status(self.status_enum.SUCCESS)

    def set_running(self):
        self.set_status(self.status_enum.RUNNING)

    def set_stopped(self, e):
        self.set_status(self.status_enum.STOPPED, e)


class ExportingInfoController(BaseInfoController):
    def __init__(self, save_path):
        super().__init__(save_path)
        self.working_info = ExportingInfo()
        self.status_enum = ExportStatus
        self.set_init()

    def set_init(self):
        self.set_status(self.status_enum.INIT)

    def set_failed(self, e):
        self.set_status(self.status_enum.FAILED, e)

    def set_success(self):
        self.working_info.step = self.working_info.total_step
        self.set_status(self.status_enum.SUCCESS)

    def set_running(self):
        self.set_status(self.status_enum.RUNNING)

    def set_stopped(self, e):
        self.set_status(self.status_enum.STOPPED, e)
