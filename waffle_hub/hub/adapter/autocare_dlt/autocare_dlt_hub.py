"""
Tx Model Hub
See Hub documentation for more details about usage.
"""

import warnings
from pathlib import Path
from typing import Union

import autocare_dlt
import tbparse
import torch
from autocare_dlt.core.model import build_model
from autocare_dlt.tools import train
from box import Box
from torchvision import transforms as T
from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_hub import TaskType
from waffle_hub.hub import Hub
from waffle_hub.hub.adapter.autocare_dlt.configs import (
    get_data_config,
    get_model_config,
)
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import TrainConfig
from waffle_hub.utils.callback import TrainCallback

from .config import DATA_TYPE_MAP, DEFAULT_PARAMS, MODEL_TYPES, WEIGHT_PATH


class AutocareDLTHub(Hub):
    BACKEND_NAME = "autocare_dlt"
    MODEL_TYPES = MODEL_TYPES
    MULTI_GPU_TRAIN = False
    DEFAULT_PARAMS = DEFAULT_PARAMS

    DATA_TYPE_MAP = DATA_TYPE_MAP
    WEIGHT_PATH = WEIGHT_PATH

    def __init__(
        self,
        name: str,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        categories: Union[list[dict], list] = None,
        root_dir: str = None,
        backend: str = None,
        version: str = None,
        *args,
        **kwargs,
    ):
        if backend is not None and AutocareDLTHub.BACKEND_NAME != backend:
            raise ValueError(
                f"Backend {backend} is not supported. Please use {AutocareDLTHub.BACKEND_NAME}"
            )

        if version is not None and autocare_dlt.__version__ != version:
            warnings.warn(
                f"You've loaded the Hub created with autocare_dlt=={version}, \n"
                + f"but the installed version is {autocare_dlt.__version__}."
            )

        super().__init__(
            name=name,
            backend=AutocareDLTHub.BACKEND_NAME,
            version=autocare_dlt.__version__,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
            root_dir=root_dir,
        )

    @classmethod
    def new(
        cls,
        name: str,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        categories: Union[list[dict], list] = None,
        root_dir: str = None,
        *args,
        **kwargs,
    ):
        """Create Tx Model Hub.

        Args:
            name (str): Hub name
            task (str, optional): Task Name. See UltralyticsHub.TASKS. Defaults to None.
            model_type (str, optional): Model Type. See UltralyticsHub.MODEL_TYPES. Defaults to None.
            model_size (str, optional): Model Size. See UltralyticsHub.MODEL_SIZES. Defaults to None.
            categories (Union[list[dict], list]): class dictionary or list. [{"supercategory": "name"}, ] or ["name",].
            root_dir (str, optional): Root directory of hub repository. Defaults to None.
        """

        warnings.warn("UltralyticsHub.new() is deprecated. Please use Hub.new() instead.")

        return cls(
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
            root_dir=root_dir,
        )

    @property
    def categories(self) -> list[dict]:
        return self.__categories

    @categories.setter
    @type_validator(list)
    def categories(self, v):
        if isinstance(v[0], str):
            v = [{"supercategory": "object", "name": n} for n in v]
        elif isinstance(v[0], dict) and "supercategory" not in v[0]:
            # TODO: Temporal solution for DLT classification: Not supported multi-task yet.
            v_ = []
            for k, cls in v[0].items():
                for c in cls:
                    v_.append({"supercategory": k, "name": c})
            v = v_
        self.__categories = v

    # Hub Utils
    def get_preprocess(self, *args, **kwargs):

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

    def get_postprocess(self, *args, **kwargs):

        if self.task == TaskType.OBJECT_DETECTION:

            def inner(x: torch.Tensor, *args, **kwargs):
                xyxy = x[0]
                scores = x[1]
                class_ids = x[2]

                return xyxy, scores, class_ids

        elif self.task == TaskType.CLASSIFICATION:

            def inner(x: torch.Tensor, *args, **kwargs):
                x = [t.squeeze() for t in x]
                return x

        elif self.task == TaskType.TEXT_RECOGNITION:

            def inner(x: torch.Tensor, *args, **kwargs):
                scores, character_class_ids = x.max(dim=-1)
                return character_class_ids, scores

        else:
            raise NotImplementedError(f"task {self.task} is not supported yet")

        return inner

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
    def on_train_start(self, cfg: TrainConfig):
        # set data
        cfg.dataset_path: Path = Path(cfg.dataset_path)
        data_config = get_data_config(
            self.DATA_TYPE_MAP[self.task],
            cfg.image_size if isinstance(cfg.image_size, list) else [cfg.image_size, cfg.image_size],
            cfg.batch_size,
            cfg.workers,
            str(cfg.dataset_path / "train.json"),
            str(cfg.dataset_path / "images"),
            str(cfg.dataset_path / "val.json"),
            str(cfg.dataset_path / "images"),
            str(cfg.dataset_path / "test.json"),
            str(cfg.dataset_path / "images"),
        )
        if self.model_type == "LicencePlateRecognition":
            data_config["data"]["mode"] = "lpr"

        cfg.data_config = self.artifact_dir / "data.json"
        io.save_json(data_config, cfg.data_config, create_directory=True)
        categories = (
            self.categories
            if self._Hub__task == TaskType.CLASSIFICATION
            else [x["name"] for x in self.categories]
        )

        model_config = get_model_config(
            self.model_type,
            self.model_size,
            categories,
            cfg.seed,
            cfg.learning_rate,
            cfg.letter_box,
            cfg.epochs,
        )

        cfg.model_config = self.artifact_dir / "model.json"
        io.save_json(model_config, cfg.model_config, create_directory=True)

        # pretrained model
        cfg.pretrained_model = (
            cfg.pretrained_model
            if cfg.pretrained_model is not None
            else self.WEIGHT_PATH[self.task][self.model_type][self.model_size]
        )
        if not Path(cfg.pretrained_model).exists():
            cfg.pretrained_model = None
            warnings.warn(f"{cfg.pretrained_model} does not exists. Train from scratch.")

        cfg.dataset_path = str(cfg.dataset_path.absolute())

    def training(self, cfg: TrainConfig, callback: TrainCallback):
        results = train.run(
            exp_name="train",
            model_cfg=str(cfg.model_config),
            data_cfg=str(cfg.data_config),
            gpus="-1" if cfg.device == "cpu" else str(cfg.device),
            output_dir=str(self.artifact_dir),
            ckpt=cfg.pretrained_model,
            overwrite=True,
        )
        if results is None:
            raise RuntimeError("Training failed")
        del results

    def on_train_end(self, cfg: TrainConfig):
        io.copy_file(
            self.artifact_dir / "train" / "best_ckpt.pth",
            self.best_ckpt_file,
            create_directory=True,
        )
        io.copy_file(
            self.artifact_dir / "train" / "last_epoch_ckpt.pth",
            self.last_ckpt_file,
            create_directory=True,
        )
        io.save_json(self.get_metrics(), self.metric_file)

    # Inference Hook
    def get_model(self):
        """Get model.
        Returns:
            ModelWrapper: Model wrapper
        """
        self.check_train_sanity()

        # get adapt functions
        preprocess = self.get_preprocess()
        postprocess = self.get_postprocess()

        # get model
        categories = [x["name"] for x in self.categories]
        cfg = io.load_json(self.artifact_dir / "model.json")
        cfg["ckpt"] = str(self.best_ckpt_file)
        if self.task == TaskType.TEXT_RECOGNITION:
            cfg["model"]["Prediction"]["num_classes"] = len(categories) + 1
        else:
            cfg["model"]["head"]["num_classes"] = len(categories)
        cfg["num_classes"] = len(categories)
        model, categories = build_model(Box(cfg), strict=True)

        model = ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
        )

        return model
