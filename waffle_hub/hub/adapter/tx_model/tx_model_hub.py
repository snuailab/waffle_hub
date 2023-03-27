"""
!!! DEPRECATED !!!
Tx Model Hub
"""

from waffle_hub import get_installed_backend_version

BACKEND_NAME = "autocare_tx_model"
BACKEND_VERSION = get_installed_backend_version(BACKEND_NAME)

from dataclasses import asdict
from pathlib import Path
from typing import Union
import tempfile

import torch
from torchvision import transforms as T

from autocare_tx_model.tools import train

from waffle_utils.file import io

from waffle_hub.utils.image import ImageDataset

from waffle_hub.hub.base_hub import BaseHub, InferenceContext, TrainContext
from waffle_hub.hub.model.wrapper import ModelWrapper, ResultParser, get_parser

from waffle_hub.hub.adapter.tx_model.configs import get_data_config, get_model_config


class TxModelHub(BaseHub):

    # Common
    MODEL_TYPES = {
        "object_detection": {
            "YOLOv5": list("sml")
        },
        # "classification": {
        #     "resnet": list("sml"),
        #     "swin": list("sml")
        # },
    }

    # Backend Specifics
    DATA_TYPE_MAP = {
        "object_detection": "COCODetectionDataset",
    }

    WEIGHTS_PATH = 

    def __init__(
        self,
        name: str,
        backend: str = None,
        version: str = None,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        classes: Union[list[dict], list] = None,
        root_dir: str = None,
    ):
        """Create Tx Model Hub.

        Args:
            name (str): Hub name
            backend (str, optional): Backend name. See waffle_hub.get_backends(). Defaults to None.
            version (str, optional): Version. See waffle_hub.get_installed_backend_version(backend). Defaults to None.
            task (str, optional): Task Name. See UltralyticsHub.TASKS. Defaults to None.
            model_type (str, optional): Model Type. See UltralyticsHub.MODEL_TYPES. Defaults to None.
            model_size (str, optional): Model Size. See UltralyticsHub.MODEL_SIZES. Defaults to None.
            classes (Union[list[dict], list]): class dictionary or list. [{"supercategory": "name"}, ] or ["name",].
            root_dir (str, optional): Root directory of hub repository. Defaults to None.
        """
        super().__init__(
            name=name,
            backend=backend if backend else BACKEND_NAME,
            version=version if version else BACKEND_VERSION,
            task=task,
            model_type=model_type,
            model_size=model_size,
            classes=classes,
            root_dir=root_dir,
        )

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
            str(ctx.dataset_path / "images")
        )
        ctx.data_config = self.artifact_dir / "data.json"
        io.save_json(data_config, ctx.data_config)

        model_config = get_model_config(
            self.model_type,
            self.model_size,
            [x["name"] for x in self.classes],
            ctx.seed,
            ctx.letter_box,
            ctx.epochs
        )
        ctx.model_config = self.artifact_dir / "model.json"
        io.save_json(model_config, ctx.model_config)

        # pretrained model
        # TODO: get pretrained model

    def training(self, ctx: TrainContext):

        train.run(
            exp_name="train",
            model_cfg=str(ctx.model_config),
            data_cfg=str(ctx.data_config),
            gpus=ctx.device,
            output_dir=str(self.artifact_dir),
        )
    