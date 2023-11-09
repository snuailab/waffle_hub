from dataclasses import dataclass

from waffle_hub import (
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
    def __init__(self, save_path):
        self.save_path = save_path

    def save(self):
        self.info.save_json(save_path=self.save_path)

    def set_total_step(self, total_step):
        self.info.total_step = total_step
        self.info.step = 0
        self.save()

    def set_current_step(self, step):
        self.info.step = step
        self.save()

    def set_failed(self, e):
        self.info.status = self.status_enum.FAILED
        self.info.error_type = e.__class__.__name__
        self.info.error_msg = e
        self.save()

    def set_success(self):
        self.info.status = self.status_enum.SUCCESS
        self.info.step = self.info.total_step
        self.info.error_type = None
        self.info.error_msg = None
        self.save()

    def set_running(self):
        self.info.status = self.status_enum.RUNNING
        self.save()

    def set_stopped(self, e):
        self.info.status = self.status_enum.STOPPED
        self.info.error_type = e.__class__.__name__
        self.info.error_msg = e
        self.save()


class TrainingInfoController(BaseInfoController):
    def __init__(self, save_path):
        super().__init__(save_path)
        self.info = TrainingInfo()
        self.status_enum = TrainStatus
        self.info.status = self.status_enum.INIT
        self.save()


class EvaluatingInfoController(BaseInfoController):
    def __init__(self, save_path):
        super().__init__(save_path)
        self.info = EvaluatingInfo()
        self.status_enum = EvaluateStatus
        self.info.status = self.status_enum.INIT
        self.save()


class InferencingInfoController(BaseInfoController):
    def __init__(self, save_path):
        super().__init__(save_path)
        self.info = InferencingInfo()
        self.status_enum = InferenceStatus
        self.info.status = self.status_enum.INIT
        self.save()


class ExportingInfoController(BaseInfoController):
    def __init__(self, save_path):
        super().__init__(save_path)
        self.info = ExportingInfo()
        self.status_enum = ExportStatus
        self.info.status = self.status_enum.INIT
        self.save()
