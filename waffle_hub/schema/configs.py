from dataclasses import dataclass
from typing import Union

from optuna.pruners import HyperbandPruner, NopPruner
from optuna.samplers import GridSampler, RandomSampler, TPESampler

from waffle_hub import HPOMethod
from waffle_hub.schema.base_schema import BaseHPOSchema, BaseSchema


@dataclass
class ModelConfig(BaseSchema):
    name: str = None
    backend: str = None
    version: str = None
    task: str = None
    model_type: str = None
    model_size: str = None
    categories: list = None


@dataclass
class TrainConfig(BaseSchema):
    dataset_path: str = None
    epochs: int = None
    batch_size: int = None
    image_size: list[int] = None
    learning_rate: float = None
    letter_box: bool = None
    pretrained_model: str = None
    device: str = None
    workers: int = None
    seed: int = None
    advance_params: dict = None
    verbose: bool = None


@dataclass
class EvaluateConfig(BaseSchema):
    dataset_name: str = None
    set_name: str = None
    batch_size: int = None
    image_size: list[int] = None
    letter_box: bool = None
    confidence_threshold: float = None
    iou_threshold: float = None
    half: bool = None
    workers: int = None
    device: str = None
    draw: bool = None
    dataset_root_dir: str = None


@dataclass
class InferenceConfig(BaseSchema):
    source: str = None
    source_type: str = None
    batch_size: int = None
    recursive: bool = None
    image_size: list[int] = None
    letter_box: bool = None
    confidence_threshold: float = None
    iou_threshold: float = None
    half: bool = None
    workers: int = None
    device: str = None
    draw: bool = None
    show: bool = None


@dataclass
class ExportConfig(BaseSchema):
    image_size: Union[int, list[int]] = None
    batch_size: int = None
    input_name: list[str] = None
    output_name: list[str] = None
    opset_version: int = None
    half: bool = False
    device: str = None


@dataclass
class HPOConfig(BaseSchema):
    dataset_path: str = None
    epochs: int = None
    batch_size: int = None
    image_size: list[int] = None
    learning_rate: list[float] = None
    letter_box: bool = None
    pretrained_model: str = None
    device: str = None
    workers: int = None
    seed: int = None
    advance_params: dict = None
    verbose: bool = None


# TODO: Whenever a new combination of sampler and pruner (i.e., method) is added,
# each framework-specific config needs to be modified separately. (Issue)


class OptunaHpoMethodConfig(BaseHPOSchema):
    framework = "OPTUNA"

    def __init__(cls):
        super().__init__(cls.framework)

    def initialize_method(self, method_type):
        if self.framework == "OPTUNA":
            method_type = method_type.upper()
            if method_type == HPOMethod.RANDOMSAMPLER.name:
                return (RandomSampler(), NopPruner())
            elif method_type == HPOMethod.GRIDSAMPLER.name:
                return (GridSampler(), NopPruner())
            elif method_type == HPOMethod.BOHB.name:
                return (TPESampler(n_startup_trials=10), HyperbandPruner())
            elif method_type == HPOMethod.TPESAMPLER.name:
                return (TPESampler(n_startup_trials=10), NopPruner())
            else:
                raise ValueError(f"Invalid sampler_type: {method_type}")
        else:
            raise ValueError("Framework mismatch")


class RaytuneHpoMethodConfig(BaseHPOSchema):
    framework = "RAYTUNE"

    def __init__(cls):
        super().__init__(cls.framework)

    def initialize_sampler(self, method_type):
        if self.framework == "RAYTUNE":
            # TODO : initialize raytune methods
            pass
        else:
            raise ValueError("Framework mismatch")
