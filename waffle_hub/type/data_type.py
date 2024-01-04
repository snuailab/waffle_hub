from waffle_utils.enum import StrEnum as BaseType


class DataType(BaseType):
    COCO = "coco"
    YOLO = "ultralytics"
    ULTRALYTICS = "ultralytics"
    AUTOCARE_DLT = "autocare_dlt"
    TRANSFORMERS = "transformers"
