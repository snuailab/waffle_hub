import os
import warnings
from pathlib import Path
from typing import Union

import autocare_dlt
import tbparse
import torch
from autocare_dlt.core.model import build_model
from autocare_dlt.tools import train
from box import Box
from torch import nn
from torchvision import transforms as T
from waffle_utils.file import io

from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.hub.train.adapter.base_manager import BaseManager
from waffle_hub.schema.fields.category import Category
from waffle_hub.type import BackendType, DataType, TaskType

from .config import DATA_TYPE_MAP, DEFAULT_PARAMS, MODEL_TYPES, WEIGHT_PATH


class AutocareDltManager(BaseManager):
    """
    Ultralytics Training Manager
    """

    BACKEND_NAME = str(BackendType.AUTOCARE_DLT.value)
    VERSION = "autocare-dlt"
    MULTI_GPU_TRAIN = False
    MODEL_TYPES = MODEL_TYPES
    DEFAULT_PARAMS = DEFAULT_PARAMS
    DEFAULT_ADVANCE_PARAMS = {}

    DATA_TYPE_MAP = DATA_TYPE_MAP
    WEIGHT_PATH = WEIGHT_PATH

    def __init__(
        self,
        root_dir: Path,
        name: str,
        task: Union[str, TaskType],
        model_type: str,
        model_size: str,
        categories: list[Union[str, int, float, dict, Category]],
        load: bool = False,
    ):
        super().__init__(
            root_dir=root_dir,
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
            load=load,
        )

        if self.VERSION is not None and autocare_dlt.__version__ != self.VERSION:
            warnings.warn(
                f"You've loaded the Hub created with autocare_dlt=={self.VERSION}, \n"
                + f"but the installed version is {autocare_dlt.__version__}."
            )

    def get_model(self) -> ModelWrapper:
        """Get model.
        Returns:
            ModelWrapper: Model wrapper
        """
        self.check_train_sanity()

        # get adapt functions
        preprocess = self.get_preprocess()
        postprocess = self.get_postprocess()

        # get model
        categories = [x["name"] for x in self.get_categories()]
        cfg = io.load_json(self.model_config_file)
        cfg["ckpt"] = str(self.best_ckpt_file)
        if self.task == TaskType.TEXT_RECOGNITION:
            cfg["model"]["Prediction"]["num_classes"] = len(categories) + 1
        else:
            cfg["model"]["head"]["num_classes"] = len(categories)
        cfg["num_classes"] = len(categories)
        model, categories = build_model(Box(cfg), strict=True)

        # return model wrapper
        return ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
        )

    def _get_preprocess(self, *args, **kwargs):

        if self.task == TaskType.OBJECT_DETECTION:
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x, *args, **kwargs):
                return normalize(x)

        elif self.task == TaskType.CLASSIFICATION:
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x, *args, **kwargs):
                return normalize(x)

        elif self.task == TaskType.TEXT_RECOGNITION:
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x, *args, **kwargs):
                return normalize(x)

        else:
            raise NotImplementedError(f"task {self.task} is not supported yet")

        return preprocess

    def _get_postprocess(self, *args, **kwargs):

        if self.task == TaskType.OBJECT_DETECTION:

            def inner(x: torch.Tensor, *args, **kwargs):
                xyxy = x[0]
                scores = x[1]
                class_ids = x[2]

                return xyxy, scores, class_ids

        elif self.task == TaskType.CLASSIFICATION:

            def inner(x: torch.Tensor, *args, **kwargs):
                x = [t.squeeze(-1).squeeze(-1) for t in x]
                return x

        elif self.task == TaskType.TEXT_RECOGNITION:

            def inner(x: torch.Tensor, *args, **kwargs):
                scores, character_class_ids = x.max(dim=-1)
                return character_class_ids, scores

        else:
            raise NotImplementedError(f"task {self.task} is not supported yet")

        return inner

    # Trainer
    def get_metrics(self) -> list[dict]:
        """Get metrics from tensorboard log directory.
        Args:
            tbdir (Union[str, Path]): tensorboard log directory
        Returns:
            list[dict]: list of metrics
        """
        tb_log_dir = self.artifact_dir / "train" / "tensorboard"
        if not tb_log_dir.exists():
            return []

        sr = tbparse.SummaryReader(tb_log_dir)
        df = sr.scalars

        # Sort the data frame by step.
        # Make a list of dictionaries of tag and value.
        if df.empty:
            return []

        metrics = (
            df.sort_values("step")
            .groupby("step")
            .apply(lambda x: [{"tag": s, "value": v} for s, v in zip(x.tag, x.value)])
            .to_list()
        )

        return metrics

    # Train Hook
    def on_train_start(self):

        # set data
        self.train_cfg.dataset_path: Path = Path(self.train_cfg.dataset_path)
        if self.backend_task_name in ["detect", "segment"]:
            if self.train_cfg.dataset_path.suffix not in [".yml", ".yaml"]:
                yaml_files = list(self.train_cfg.dataset_path.glob("*.yaml")) + list(
                    self.train_cfg.dataset_path.glob("*.yml")
                )
                if len(yaml_files) != 1:
                    raise FileNotFoundError(f"Ambiguous data file. Detected files: {yaml_files}")
                self.train_cfg.dataset_path = Path(yaml_files[0]).absolute()
            else:
                self.train_cfg.dataset_path = self.train_cfg.dataset_path.absolute()
        elif self.backend_task_name == "classify":
            if not self.train_cfg.dataset_path.is_dir():
                raise ValueError(
                    f"Classification dataset should be directory. Not {self.train_cfg.dataset_path}"
                )
            self.train_cfg.dataset_path = self.train_cfg.dataset_path.absolute()
        self.train_cfg.dataset_path = str(self.train_cfg.dataset_path)

        # pretrained model
        self.train_cfg.pretrained_model = (
            self.train_cfg.pretrained_model
            if self.train_cfg.pretrained_model
            else self.MODEL_TYPES[self.task][self.model_type][self.model_size]
        )

        # other
        if self.task in [TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]:
            self.train_cfg.letter_box = True  # TODO: hard coding
            # logger.warning(
            #     "letter_box False is not supported for Object Detection and Segmentation."
            # )

    def training(self):
        params = {
            "data": str(self.train_cfg.dataset_path).replace("\\", "/"),
            "epochs": self.train_cfg.epochs,
            "batch": self.train_cfg.batch_size,
            "imgsz": self.train_cfg.image_size,
            "lr0": self.train_cfg.learning_rate,
            "lrf": self.train_cfg.learning_rate,
            "rect": False,  # TODO: hard coding for mosaic
            "device": self.train_cfg.device,
            "workers": self.train_cfg.workers,
            "seed": self.train_cfg.seed,
            "verbose": self.train_cfg.verbose,
            "project": str(self.root_dir),
            "name": str(self.ARTIFACTS_DIR),
        }
        params.update(self.train_cfg.advance_params)

        code = f"""if __name__ == "__main__":
        from ultralytics import YOLO
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        try:
            model = YOLO("{self.train_cfg.pretrained_model}", task="{self.backend_task_name}")
            model.train(
                **{params}
            )
        except Exception as e:
            print(e)
            raise e
        """

        script_file = str((self.root_dir / "train.py").absolute())
        with open(script_file, "w") as f:
            f.write(code)

        run_python_file(script_file)

    def on_train_end(self):
        io.copy_file(
            self.artifacts_dir / "weights" / "best.pt",
            self.best_ckpt_file,
            create_directory=True,
        )
        io.copy_file(
            self.artifacts_dir / "weights" / "last.pt",
            self.last_ckpt_file,
            create_directory=True,
        )
        io.save_json(self.get_metrics(), self.metric_file)
