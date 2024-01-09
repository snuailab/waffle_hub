from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from waffle_hub import (
    BaseStatus,
    EvaluateStatus,
    ExportOnnxStatus,
    ExportWaffleStatus,
    InferenceStatus,
    TrainStatus,
)
from waffle_hub.schema.base_schema import BaseSchema

STATUS_TYPE = TypeVar("STATUS_TYPE", bound=BaseStatus)


@dataclass
class BaseState(Generic[STATUS_TYPE], BaseSchema):
    status: STATUS_TYPE
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    step: Optional[int] = None
    total_step: Optional[int] = None

    def __setattr__(self, name, value):
        if name == "status":
            if not (value is None or value in self.__class__.__orig_bases__[0].__args__[0]):
                raise ValueError(
                    f"status must be one of {self.__class__.__orig_bases__[0].__args__[0]}"
                )
        self.__dict__[name] = value

    def set_error(self, e: Exception):
        self.error_type = e.__class__.__name__
        self.error_msg = e

    def clear_error(self):
        self.error_type = None
        self.error_msg = None

    @classmethod
    def from_dict(cls, d):
        temp_cls = cls(cls.__orig_bases__[0].__args__[0]("init"))
        for k, v in d.items():
            if k == "status":
                v = cls.__orig_bases__[0].__args__[0](v)
            setattr(temp_cls, k, v)
        return temp_cls


@dataclass
class TrainState(BaseState[TrainStatus]):
    pass


@dataclass
class EvaluateState(BaseState[EvaluateStatus]):
    pass


@dataclass
class InferenceState(BaseState[InferenceStatus]):
    pass


@dataclass
class ExportOnnxState(BaseState[ExportOnnxStatus]):
    pass


@dataclass
class ExportWaffleState(BaseState[ExportWaffleStatus]):
    pass
