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


def get_preprocess(task: str, *args, **kwargs):

    if task == "object_detection":
        normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

        def preprocess(x):
            return normalize(x)

    return preprocess


def get_postprocess(task: str, *args, **kwargs):

    if task == "object_detection":

        def inner(x: torch.Tensor):
            return x

    return inner


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
            str(ctx.dataset_path / "images"),
        )
        ctx.data_config = self.artifact_dir / "data.json"
        io.save_json(data_config, ctx.data_config, create_directory=True)

        model_config = get_model_config(
            self.model_type,
            self.model_size,
            [x["name"] for x in self.classes],
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

    def training(self, ctx: TrainContext):

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

    # Inference Hook
    def get_model(
        self, image_size: Union[int, list] = None, parser: ResultParser = None
    ):
        """Get model.
        Args:
            image_size (Union[int, list], optional): Image size. Defaults to None.
            parser (ResultParser, optional): Result parser. Defaults to None.
        Returns:
            ModelWrapper: Model wrapper
        """
        self.check_train_sanity()

        # get adapt functions
        preprocess = get_preprocess(self.task)
        postprocess = get_postprocess(self.task)

        # get model
        classes = [x["name"] for x in self.classes]
        cfg = io.load_json(self.artifact_dir / "model.json")
        cfg["model"]["head"]["num_classes"] = len(classes)
        cfg["ckpt"] = str(self.best_ckpt_file)
        cfg["classes"] = classes
        cfg["num_classes"] = (len(classes),)
        model, classes = build_model(AttrDict(cfg), strict=True)

        model = ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
            parser=parser if parser else None,
        )

        return model
