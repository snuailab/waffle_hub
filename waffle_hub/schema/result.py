from dataclasses import dataclass

from waffle_hub.schema.base_schema import BaseSchema


@dataclass
class BaseResult(BaseSchema):
    pass


@dataclass
class TrainResult(BaseResult):
    best_ckpt_file: str = None
    last_ckpt_file: str = None
    metrics: list[list[dict]] = None
    eval_metrics: list[dict] = None


@dataclass
class EvaluateResult(BaseResult):
    eval_metrics: list[dict] = None


@dataclass
class InferenceResult(BaseResult):
    predictions: list[dict[list]] = None
    draw_dir: str = None


@dataclass
class ExportOnnxResult(BaseResult):
    onnx_file: str = None


@dataclass
class ExportWaffleResult(BaseResult):
    waffle_file: str = None
