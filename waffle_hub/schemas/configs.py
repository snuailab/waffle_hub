from dataclasses import dataclass


@dataclass
class Model:
    name: str
    backend: str
    version: str
    task: str
    model_type: str
    model_size: str


@dataclass
class Train:
    image_size: int
    batch_size: int
    pretrained_model: str
    seed: int


@dataclass
class Classes:
    names: list[str]


# TODO: Should be moved to waffle_utils. Define Prediction schema for now.
@dataclass
class Prediction:
    image_path: str
    predictions: list[dict]


@dataclass
class DetectionPrediction:
    bbox: list[float]
    class_name: str
    confidence: float
    segment: list[float] = list


@dataclass
class ClassificationPrediction:
    score: float
    class_name: str
