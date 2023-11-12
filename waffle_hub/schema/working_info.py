from dataclasses import dataclass
from typing import Generic, TypeVar

from waffle_hub import (
    BaseEnum,
    EvaluateStatus,
    ExportStatus,
    InferenceStatus,
    TrainStatus,
)
from waffle_hub.schema.base_schema import BaseSchema

STATUS_TYPE = TypeVar("STATUS_TYPE", bound=BaseEnum)


@dataclass
class BaseWorkingInfo(Generic[STATUS_TYPE], BaseSchema):
    status: STATUS_TYPE = None
    error_type: str = None
    error_msg: str = None
    step: int = None
    total_step: int = None

    def __setattr__(self, name, value):
        if name == "status":
            if not (value is None or value in self.__class__.__orig_bases__[0].__args__[0]):
                raise ValueError(
                    f"status must be one of {self.__class__.__orig_bases__[0].__args__[0]}"
                )
        self.__dict__[name] = value


@dataclass
class TrainingInfo(BaseWorkingInfo[TrainStatus]):
    pass


@dataclass
class EvaluatingInfo(BaseWorkingInfo[EvaluateStatus]):
    pass


@dataclass
class InferencingInfo(BaseWorkingInfo[InferenceStatus]):
    pass


@dataclass
class ExportingInfo(BaseWorkingInfo[ExportStatus]):
    pass
