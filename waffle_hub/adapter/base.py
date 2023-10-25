from abc import ABC, abstractmethod

from torch import nn


class Model(ABC):  # one for each backend
    """
    1. define model
    2. crud
    """

    def __init__(
        self,
        model_type: str,
        model_size: str,
        cateogories: list[str],
    ):
        self.model = ""

    @property
    def model(self) -> nn.Module:
        pass

    @property
    @model.setter
    def model(self, v):
        if not isinstance(v, nn.Module):
            raise

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass

    def save(self, model_path):
        pass

    def load(self, model_path):
        pass


class Trainer(ABC):  # one for each backend
    """
    1. train
    2. resume
    """

    def __init__(
        self,
        model_dir,
    ):
        pass

    def train(
        self,
        epochs: int,
        batch_size: int,
        train_dir: str,
        dataset_path: str,
    ):
        self.before_train()
        self.training()
        self.after_train()

    def before_train(self):
        """
        1. configuration
        """
        pass

    def training(self):
        pass

    def after_train(self):
        """
        1. save results
        """
        pass


class Ultralytics(ModelWrapper, Trainer):
    BEST_CKPT_FILE = "best.pth"

    @property
    def best_ckpt_file(self) -> str:
        return Path(self.model_dir, self.BEST_CKPT_FILE)

    def __init__(self, model_type: str, model_size: str, cateogories: list[str]):
        super(ModelWrapper).__init__(model_type, model_size, cateogories)
        super(Trainer).__init__

    def train(self):
        pass

    def save(self):
        pass

    def load(self, model_path):
        pass


# class ModelManager:
#     def __init__(self, model: ModelWrapper, trainer: Trainer):
#         pass

#     def train(self):
#         pass

#     def get_model(self):
#         pass


class Hub:
    def __init__(self) -> None:
        self.model = Ultralytics()
        pass

    @property
    def best_model_path(self):
        return self.model.best_ckpt_file

    def train(self):

        self.model.train(train_dir=self.train_dir, **train_cfg)

    def inference(self):
        model = Ultralytics().get_model()
