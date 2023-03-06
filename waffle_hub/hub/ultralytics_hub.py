import warnings

from waffle_hub import get_backends

try:
    import ultralytics

    versions = get_backends()["ultralytics"]
    if ultralytics.__version__ not in versions:
        warnings.warn(
            f"""
            ultralytics {ultralytics.__version__} has not been tested.
            We recommend you to use one of {versions}
            """
        )
except ModuleNotFoundError as e:
    versions = get_backends()["ultralytics"]

    strings = [f"- pip install ultralytics=={version}" for version in versions]

    e.msg = "Need to install ultralytics\n" + "\n".join(strings)
    raise e


from pathlib import Path

from pydantic import BaseModel
from ultralytics import YOLO
from waffle_utils.file import io

from . import BaseHub


class BaseModelParams(BaseModel):
    model_name: str
    image_size: str


class UltralyticsHub(BaseHub):

    # Common
    AVAILABLE_TASK = ["detect", "classify", "segment"]
    AVAILABLE_MODEL = ["yolov8"]
    AVAILABLE_SIZE = list("nsmlx")

    # Backend Specifics
    TASK_SUFFIX = {
        "detect": "",
        "classify": "-cls",
        "segment": "-seg",
    }

    def __init__(
        self,
        name: str,
        task: str = None,
        model_name: str = None,
        model_size: str = None,
        pretrained_model: str = None,
        root_dir: str = None,
    ):

        super().__init__(
            name=name,
            task=task,
            model_name=model_name,
            model_size=model_size,
            pretrained_model=pretrained_model,
            root_dir=root_dir,
        )

        # ultralytics specific variables
        self._model = (
            self.model_name
            + self.model_size
            + self.TASK_SUFFIX[self.task]
            + ".pt"
            if pretrained_model is None
            else pretrained_model
        )

    def validate_model(self):
        return

    def train(
        self,
        dataset_dir: str,
        epochs: int,
        batch_size: int,
        image_size: int,
        device: str = "0",
        workers: int = 2,
        seed: int = 0,
        verbose: bool = True,
    ):
        self.is_trainable()

        # set data
        dataset_dir: Path = Path(dataset_dir)
        if self.task in ["detect", "segment"]:
            if dataset_dir.suffix not in [".yml", ".yaml"]:
                yaml_files = list(dataset_dir.glob("*.yaml")) + list(
                    dataset_dir.glob("*.yml")
                )
                if len(yaml_files) != 1:
                    raise FileNotFoundError(
                        f"Ambiguous data file. Detected files: {yaml_files}"
                    )
                data = Path(yaml_files[0]).absolute()
            else:
                data = dataset_dir.absolute()
        elif self.task == "classify":
            if not dataset_dir.is_dir():
                raise ValueError(
                    f"Classification dataset should be directory. Not {dataset_dir}"
                )
            data = dataset_dir.absolute()
        data = str(data)

        # save train configs. TODO: Generalize using pydantic
        io.save_yaml(
            {
                "task": self.task,
                "model_name": self.model_name,
                "model_size": self.model_size,
                "pretrained_model": self.pretrained_model
                if self.pretrained_model
                else self._model,
                "image_size": image_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
            },
            self.train_option_file,
            create_directory=True,
        )

        try:
            model = YOLO(self._model, task=self.task)
            model.train(
                data=data,
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                device=device,
                workers=workers,
                seed=seed,
                verbose=verbose,
                project=self.model_dir,
                name=self.RAW_TRAIN_DIR,
            )

            # Parse Training Results
            io.copy_file(
                self.raw_train_dir / "weights" / "best.pt",
                self.best_ckpt_file,
                create_directory=True,
            )
            io.copy_file(
                self.raw_train_dir / "weights" / "last.pt",
                self.last_ckpt_file,
                create_directory=True,
            )
            io.copy_file(
                self.raw_train_dir / "results.csv",
                self.metric_file,
                create_directory=True,
            )

            return str(self.train_dir)

        except Exception as e:
            if self.raw_train_dir.exists():
                io.remove_directory(self.raw_train_dir)
            raise e

    def inference(self):
        raise NotImplementedError

    def evaluation(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
