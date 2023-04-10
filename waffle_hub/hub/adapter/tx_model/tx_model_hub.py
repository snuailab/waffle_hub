"""
!!! DEPRECATED !!!
Tx Model Hub
"""

from waffle_hub import get_installed_backend_version

BACKEND_NAME = "autocare_tx_model"
BACKEND_VERSION = get_installed_backend_version(BACKEND_NAME)

import warnings
from pathlib import Path
from typing import Union

import tbparse
import torch
from attrdict import AttrDict
from autocare_tx_model.core.model import build_model
from autocare_tx_model.tools import train
from torchvision import transforms as T
from waffle_utils.file import io

from waffle_hub.hub.adapter.tx_model.configs import (
    get_data_config,
    get_model_config,
)
from waffle_hub.hub.base_hub import BaseHub, TrainContext
from waffle_hub.hub.model.wrapper import ModelWrapper, ResultParser
from waffle_hub.utils.callback import TrainCallback


class TxModelHub(BaseHub):

    # Common
    MODEL_TYPES = {
        "object_detection": {"YOLOv5": list("sml")},
        # "classification": {
        #     "resnet": list("sml"),
        #     "swin": list("sml")
        # },
    }

    # Backend Specifics
    DATA_TYPE_MAP = {
        "object_detection": "COCODetectionDataset",
    }

    WEIGHT_PATH = {
        "object_detection": {
            "YOLOv5": {
                "s": "temp/autocare_tx_model/detectors/small/model.pth",
                "m": "temp/autocare_tx_model/detectors/medium/model.pth",
                "l": "temp/autocare_tx_model/detectors/large/model.pth",
            }
        },
    }

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
    ):
        """Create Tx Model Hub Class. Do not use this class directly. Use TxModelHub.new() instead."""

        if backend is not None and backend != BACKEND_NAME:
            raise ValueError(
                f"you've loaded {backend}. backend must be {BACKEND_NAME}"
            )

        if version is not None and version != BACKEND_VERSION:
            warnings.warn(
                f"you've loaded a {BACKEND_NAME}=={version} version while {BACKEND_NAME}=={BACKEND_VERSION} version is installed."
                "It will cause unexpected results."
            )

        super().__init__(
            name=name,
            backend=BACKEND_NAME,
            version=BACKEND_VERSION,
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
        return cls(
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
            root_dir=root_dir,
        )

    # Hub Utils
    def get_preprocess(self, task: str, *args, **kwargs):

        if task == "object_detection":
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x, *args, **kwargs):
                return normalize(x)

        return preprocess

    def get_postprocess(self, task: str, *args, **kwargs):

        if task == "object_detection":

            def inner(x: torch.Tensor, *args, **kwargs):
                return x

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
            .apply(
                lambda x: [
                    {"tag": s, "value": v} for s, v in zip(x.tag, x.value)
                ]
            )
            .to_list()
        )

        return metrics

    # Train Hook
    def on_train_start(self, ctx: TrainContext):
        # set data
        ctx.dataset_path: Path = Path(ctx.dataset_path)

        data_config = get_data_config(
            self.DATA_TYPE_MAP[self.task],
            [ctx.image_size, ctx.image_size],
            ctx.batch_size,
            ctx.workers,
            str(ctx.dataset_path / "train.json"),
            str(ctx.dataset_path / "images"),
            str(ctx.dataset_path / "val.json"),
            str(ctx.dataset_path / "images"),
            str(ctx.dataset_path / "test.json"),
            str(ctx.dataset_path / "images"),
        )
        ctx.data_config = self.artifact_dir / "data.json"
        io.save_json(data_config, ctx.data_config, create_directory=True)

        model_config = get_model_config(
            self.model_type,
            self.model_size,
            [x["name"] for x in self.categories],
            ctx.seed,
            ctx.letter_box,
            ctx.epochs,
        )
        ctx.model_config = self.artifact_dir / "model.json"
        io.save_json(model_config, ctx.model_config, create_directory=True)

        # pretrained model
        ctx.pretrained_model = (
            ctx.pretrained_model
            if ctx.pretrained_model is not None
            else self.WEIGHT_PATH[self.task][self.model_type][self.model_size]
        )
        if not Path(ctx.pretrained_model).exists():
            ctx.pretrained_model = None
            warnings.warn(
                f"{ctx.pretrained_model} does not exists. Train from scratch."
            )

    def training(self, ctx: TrainContext, callback: TrainCallback):

        results = train.run(
            exp_name="train",
            model_cfg=str(ctx.model_config),
            data_cfg=str(ctx.data_config),
            gpus="-1" if ctx.device == "cpu" else str(ctx.device),
            output_dir=str(self.artifact_dir),
            ckpt=ctx.pretrained_model,
            overwrite=True,
        )
        del results

    def on_train_end(self, ctx: TrainContext):
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
        preprocess = self.get_preprocess(self.task)
        postprocess = self.get_postprocess(self.task)

        # get model
        categories = [x["name"] for x in self.categories]
        cfg = io.load_json(self.artifact_dir / "model.json")
        cfg["model"]["head"]["num_categories"] = len(categories)
        cfg["ckpt"] = str(self.best_ckpt_file)
        cfg["categories"] = categories
        cfg["num_categories"] = (len(categories),)
        model, categories = build_model(AttrDict(cfg), strict=True)

        model = ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
        )

        return model
