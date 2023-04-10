from dataclasses import dataclass
from typing import Union

from waffle_hub.schema.base_schema import BaseSchema


@dataclass
class ModelConfig(BaseSchema):
    name: str = None
    backend: str = None
    version: str = None
    task: str = None
    model_type: str = None
    model_size: str = None
    classes: list = None


@dataclass
class TrainConfig(BaseSchema):
    dataset_path: str = None
    epochs: int = None
    batch_size: int = None
    image_size: Union[int, list[int]] = None
    learning_rate: float = None
    letter_box: bool = None
    pretrained_model: str = None
    device: str = None
    workers: int = None
    seed: int = None
    verbose: bool = None


@dataclass
class InferenceConfig(BaseSchema):
    source: str = None
    batch_size: int = None
    recursive: bool = None
    image_size: Union[int, list[int]] = None
    letter_box: bool = None
    confidence_threshold: float = None
    iou_threshold: float = None
    half: bool = None
    workers: int = None
    device: str = None
    draw: bool = None


@dataclass
class ExportConfig(BaseSchema):
    image_size: Union[int, list[int]] = None
    batch_size: int = None
    input_name: list[str] = None
    output_name: list[str] = None
    opset_version: int = None
