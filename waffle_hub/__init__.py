__version__ = "0.4.0a1"

import signal
from collections import OrderedDict

from waffle_utils.enum import StrEnum as BaseEnum

from waffle_hub.type import BackendType, DataType
from waffle_hub.utils.utils import CaseInsensitiveDict


class SplitMethod(BaseEnum):
    RANDOM = "random"
    STRATIFIED = "stratified"


# for changeable status desc
class BaseStatus(BaseEnum):
    pass


class TrainStatus(BaseStatus):
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


class EvaluateStatus(BaseStatus):
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


class InferenceStatus(BaseStatus):
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


class ExportOnnxStatus(BaseStatus):
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


class ExportWaffleStatus(BaseStatus):
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


EXPORT_MAP = CaseInsensitiveDict(
    {
        DataType.YOLO: "ULTRALYTICS",
        DataType.ULTRALYTICS: "ULTRALYTICS",
        DataType.COCO: "COCO",
        DataType.AUTOCARE_DLT: "AUTOCARE_DLT",
        DataType.TRANSFORMERS: "TRANSFORMERS",
    }
)

BACKEND_MAP = CaseInsensitiveDict(
    {
        BackendType.ULTRALYTICS: {
            "adapter_import_path": "waffle_hub.hub.manager.adapter.ultralytics.ultralytics",
            "adapter_class_name": "UltralyticsManager",
        },
        BackendType.AUTOCARE_DLT: {
            "adapter_import_path": "waffle_hub.hub.manager.adapter.autocare_dlt.autocare_dlt",
            "adapter_class_name": "AutocareDltManager",
        },
        BackendType.TRANSFORMERS: {
            "adapter_import_path": "waffle_hub.hub.manager.adapter.transformers.transformers",
            "adapter_class_name": "TransformersManager",
        },
    }
)


for key in list(EXPORT_MAP.keys()):
    EXPORT_MAP[str(key).lower()] = EXPORT_MAP[key]
    EXPORT_MAP[str(key).upper()] = EXPORT_MAP[key]


for key in list(BACKEND_MAP.keys()):
    BACKEND_MAP[str(key).lower()] = BACKEND_MAP[key]
    BACKEND_MAP[str(key).upper()] = BACKEND_MAP[key]
