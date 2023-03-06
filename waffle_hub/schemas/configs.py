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
