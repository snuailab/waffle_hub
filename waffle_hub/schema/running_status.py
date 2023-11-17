from dataclasses import dataclass
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

    def __setattr__(self, name, value):
        if name == "status_desc":
            if not (value is None or value in self.__class__.__orig_bases__[0].__args__[0]):
                raise ValueError(
                    f"status must be one of {self.__class__.__orig_bases__[0].__args__[0]}"
                )
        self.__dict__[name] = value


@dataclass
class TrainingStatus(BaseRunningStatus[TrainStatusDesc]):
    pass


@dataclass
class EvaluatingStatus(BaseRunningStatus[EvaluateStatusDesc]):
    pass


@dataclass
class InferencingStatus(BaseRunningStatus[InferenceStatusDesc]):
    pass


@dataclass
class ExportingOnnxStatus(BaseRunningStatus[ExportOnnxStatusDesc]):
    pass


@dataclass
class ExportingWaffleStatus(BaseRunningStatus[ExportWaffleStatusDesc]):
    pass
