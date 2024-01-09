from waffle_utils.enum import StrEnum as BaseType


class BackendType(BaseType):
    ULTRALYTICS = "ultralytics"
    AUTOCARE_DLT = "autocare_dlt"
    TRANSFORMERS = "transformers"
