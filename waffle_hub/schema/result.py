from dataclasses import dataclass

from waffle_hub.schema.base_schema import BaseSchema


@dataclass
class TrainResult(BaseSchema):
    best_ckpt_file: str = None
    last_ckpt_file: str = None
    metrics: list[list[dict]] = None
    eval_metrics: list[dict] = None


@dataclass
class EvaluateResult(BaseSchema):
    eval_metrics: list[dict] = None


@dataclass
class InferenceResult(BaseSchema):
    predictions: list[dict[list]] = None
    draw_dir: str = None


@dataclass
class ExportResult(BaseSchema):
    export_file: str = None
