from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from waffle_hub import (
    BaseEnum,
    EvaluateStatusDesc,
    ExportOnnxStatusDesc,
    ExportWaffleStatusDesc,
    InferenceStatusDesc,
    TrainStatusDesc,
)
from waffle_hub.schema.base_schema import BaseSchema

STATUS_DESC_TYPE = TypeVar("STATUS_DESC_TYPE", bound=BaseEnum)


@dataclass
class BaseRunningStatus(Generic[STATUS_DESC_TYPE], BaseSchema):
    status_desc: STATUS_DESC_TYPE = None
    error_type: str = None
    error_msg: str = None
    step: int = None
    total_step: int = None

    def __init__(self, root_dir: Path, file_name: Path):
        super().__init__()
        self.save_path = Path(root_dir) / "status" / file_name

    def __setattr__(self, name, value):
        if name == "status_desc":
            if not (value is None or value in self.__class__.__orig_bases__[0].__args__[0]):
                raise ValueError(
                    f"status must be one of {self.__class__.__orig_bases__[0].__args__[0]}"
                )
        self.__dict__[name] = value

    def save(self):
        self.save_json(save_path=self.save_path)

    def set_status(self, status: STATUS_DESC_TYPE):
        self.status_desc = status
        self.save()

    def set_error(self, e: Exception = None):
        self.error_type = e.__class__.__name__
        self.error_msg = e
        self.save()

    def clear_error(self):
        self.error_type = None
        self.error_msg = None
        self.save()

    def clear_step(self):
        self.step = 0
        self.total_step = 0
        self.save()

    def set_total_step(self, total_step: int):
        self.total_step = total_step
        self.step = 0
        self.save()

    def set_current_step(self, step: int):
        self.step = step
        self.save()


@dataclass
class TrainingStatus(BaseRunningStatus[TrainStatusDesc]):
    def __init__(self, root_dir: Path):
        super().__init__(root_dir=root_dir, file_name="training_status.json")
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
        self.step = self.total_step
        self.set_status(TrainStatusDesc.SUCCESS)

    def set_running(self):
        self.set_status(TrainStatusDesc.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(TrainStatusDesc.STOPPED)


@dataclass
class EvaluatingStatus(BaseRunningStatus[EvaluateStatusDesc]):
    def __init__(self, root_dir: Path):
        super().__init__(root_dir=root_dir, file_name="evaluating_status.json")
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
        self.step = self.total_step
        self.set_status(EvaluateStatusDesc.SUCCESS)

    def set_running(self):
        self.set_status(EvaluateStatusDesc.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(EvaluateStatusDesc.STOPPED)


@dataclass
class InferencingStatus(BaseRunningStatus[InferenceStatusDesc]):
    def __init__(self, root_dir: Path):
        super().__init__(root_dir=root_dir, file_name="inferencing_status.json")
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
        self.step = self.total_step
        self.set_status(InferenceStatusDesc.SUCCESS)

    def set_running(self):
        self.set_status(InferenceStatusDesc.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(InferenceStatusDesc.STOPPED)


@dataclass
class ExportingOnnxStatus(BaseRunningStatus[ExportOnnxStatusDesc]):
    def __init__(self, root_dir: Path):
        super().__init__(root_dir=root_dir, file_name="exporting_onnx_status.json")
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
        self.step = self.total_step
        self.set_status(ExportOnnxStatusDesc.SUCCESS)

    def set_running(self):
        self.set_status(ExportOnnxStatusDesc.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)


@dataclass
class ExportingWaffleStatus(BaseRunningStatus[ExportWaffleStatusDesc]):
    def __init__(self, root_dir: Path):
        super().__init__(root_dir=root_dir, file_name="exporting_waffle_status.json")
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
        self.step = self.total_step
        self.set_status(ExportWaffleStatusDesc.SUCCESS)

    def set_running(self):
        self.set_status(ExportWaffleStatusDesc.RUNNING)

    def set_stopped(self, e):
        self.set_error(e)
        self.set_status(ExportWaffleStatusDesc.STOPPED)
