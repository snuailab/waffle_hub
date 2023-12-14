__version__ = "0.3.0a1"

import enum
import signal
from collections import OrderedDict

from waffle_utils.enum import StrEnum as BaseEnum

from waffle_hub.type.backend_type import BackendType
from waffle_hub.type.data_type import DataType
from waffle_hub.utils.utils import CaseInsensitiveDict


class SplitMethod(BaseEnum):
    RANDOM = "random"
    STRATIFIED = "stratified"


# for changeable status desc
class TrainStatusDesc(BaseEnum):
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


class EvaluateStatusDesc(BaseEnum):
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


class InferenceStatusDesc(BaseEnum):
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


class ExportOnnxStatusDesc(BaseEnum):
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


class ExportWaffleStatusDesc(BaseEnum):
    INIT = "init"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


EXPORT_MAP = OrderedDict(
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
            "import_path": "waffle_hub.hub.adapter.ultralytics",
            "class_name": "UltralyticsHub",
        },
        BackendType.AUTOCARE_DLT: {
            "import_path": "waffle_hub.hub.adapter.autocare_dlt",
            "class_name": "AutocareDLTHub",
        },
        BackendType.TRANSFORMERS: {
            "import_path": "waffle_hub.hub.adapter.transformers",
            "class_name": "TransformersHub",
        },
    }
)
# BACKEND_MAP = CaseInsensitiveDict(
#     {
#         BackendType.ULTRALYTICS: {
#             "adapter_import_path": "waffle_hub.hub.train.adapter.ultralytics.ultralytics",
#             "adapter_class_name": "UltralyticsManager",
#         },
#         BackendType.AUTOCARE_DLT: {
#             "adapter_import_path": "waffle_hub.hub.train.adapter.autocare_dlt.autocare_dlt",
#             "adapter_class_name": "AutocareDltManager",
#         },
#         BackendType.TRANSFORMERS: {
#             "adapter_import_path": "waffle_hub.hub.train.adapter.transformers.transformers",
#             "adapter_class_name": "TransformersManager",
#         },
#     }
# )


for key in list(EXPORT_MAP.keys()):
    EXPORT_MAP[str(key).lower()] = EXPORT_MAP[key]
    EXPORT_MAP[str(key).upper()] = EXPORT_MAP[key]


for key in list(BACKEND_MAP.keys()):
    BACKEND_MAP[str(key).lower()] = BACKEND_MAP[key]
    BACKEND_MAP[str(key).upper()] = BACKEND_MAP[key]


# except handler for SIGINT, SIGTERM, SIGCHILD
def sigint_handler(signum, frame):
    raise KeyboardInterrupt


def sigterm_handler(signum, frame):
    raise SystemExit


signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigterm_handler)
