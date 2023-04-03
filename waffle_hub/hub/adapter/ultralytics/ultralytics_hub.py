"""
Ultralytics Hub
"""

from waffle_hub import get_installed_backend_version

BACKEND_NAME = "ultralytics"
BACKEND_VERSION = get_installed_backend_version(BACKEND_NAME)

import warnings
from pathlib import Path
from typing import Union

import torch
from torchvision import transforms as T
from ultralytics import YOLO
from waffle_utils.file import io

from waffle_hub.hub.base_hub import BaseHub, InferenceContext, TrainContext
from waffle_hub.hub.model.wrapper import ModelWrapper, ResultParser
from waffle_hub.utils.callback import TrainCallback


class UltralyticsHub(BaseHub):

    # Common
    MODEL_TYPES = {
        "object_detection": {"yolov8": list("nsmlx")},
        "classification": {"yolov8": list("nsmlx")},
        # "segmentation": {"yolov8": list("nsmlx")},
        # "keypoint_detection": {"yolov8": list("nsmlx")},
    }

    # Backend Specifics
    TASK_MAP = {
        "object_detection": "detect",
        "classification": "classify",
        # "segmentation": "segment"
        # "keypoint_detection": "pose"
    }
    TASK_SUFFIX = {
        "detect": "",
        "classify": "-cls",
        # "segment": "-seg",
    }

    def __init__(
        self,
        name: str,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        classes: Union[list[dict], list] = None,
        root_dir: str = None,
        backend: str = None,
        version: str = None,
    ):
        """Create Ultralytics Hub Class. Do not use this class directly. Use UltralyticsHub.new() instead."""

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
            classes=classes,
            root_dir=root_dir,
        )

        self.backend_task_name = self.TASK_MAP[self.task]

    @classmethod
    def new(
        cls,
        name: str,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        classes: Union[list[dict], list] = None,
        root_dir: str = None,
    ):
        """Create Ultralytics Hub.

        Args:
            name (str): Hub name
            task (str, optional): Task Name. See UltralyticsHub.TASKS. Defaults to None.
            model_type (str, optional): Model Type. See UltralyticsHub.MODEL_TYPES. Defaults to None.
            model_size (str, optional): Model Size. See UltralyticsHub.MODEL_SIZES. Defaults to None.
            classes (Union[list[dict], list]): class dictionary or list. [{"supercategory": "name"}, ] or ["name",].
            root_dir (str, optional): Root directory of hub repository. Defaults to None.
        """
        return cls(
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            classes=classes,
            root_dir=root_dir,
        )

    # Hub Utils
    def get_preprocess(self, task: str, *args, **kwargs):

        if task == "classification":
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x):
                return normalize(x)

        elif task == "object_detection":
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x):
                return normalize(x)

        else:
            raise NotImplementedError(f"Task {task} is not implemented.")

        return preprocess

    def get_postprocess(self, task: str, *args, **kwargs):

        if task == "classification":

            def inner(x: torch.Tensor):
                return [x]

        elif task == "object_detection":
            image_size = kwargs.get("image_size")
            image_size = (
                image_size
                if isinstance(image_size, list)
                else [image_size, image_size]
            )

            def inner(x: torch.Tensor):
                x = x[0]  # x[0]: prediction, x[1]: TODO: what is this...?
                x = x.transpose(1, 2)

                cxcywh = x[:, :, :4]
                cx, cy, w, h = torch.unbind(cxcywh, dim=-1)
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                xyxy[:, :, ::2] /= image_size[0]
                xyxy[:, :, 1::2] /= image_size[1]
                probs = x[:, :, 4:]
                confidences, class_ids = torch.max(probs, dim=-1)

                return xyxy, confidences, class_ids

        else:
            raise NotImplementedError(f"Task {task} is not implemented.")

        return inner

    def get_metrics(self) -> list[list[dict]]:
        # read csv file
        # epoch,         train/box_loss,         train/cls_loss,         train/dfl_loss,   metrics/precision(B),      metrics/recall(B),       metrics/mAP50(B),    metrics/mAP50-95(B),           val/box_loss,           val/cls_loss,           val/dfl_loss,                 lr/pg0,                 lr/pg1,                 lr/pg2
        #              0,                 2.0349,                 4.4739,                  1.214,                0.75962,                0.34442,                0.22858,                0.18438,                0.61141,                 2.8409,                0.78766,                 0.0919,                 0.0009,                 0.0009
        #              1,                 1.7328,                 4.0078,                 1.1147,                0.12267,                0.40909,                0.20891,                0.15007,                0.75082,                 2.7157,                0.79723,               0.082862,              0.0018624,              0.0018624
        # and parse it to list[dict]
        # [[{"tag": train/box_loss, "value": 0.0}, {"tag": train/box_loss, "value": 1.7328}, ...], ...]
        # and return it

        csv_path = self.artifact_dir / "results.csv"

        if not csv_path.exists():
            return []

        with open(csv_path) as f:
            lines = f.readlines()

        header = lines[0].strip().split(",")
        metrics = []
        for line in lines[1:]:
            values = line.strip().split(",")[1:]
            metric = []
            for i, value in enumerate(values):
                metric.append(
                    {
                        "tag": header[i].strip(),
                        "value": float(value),
                    }
                )
            metrics.append(metric)

        return metrics

    # Train Hook
    def on_train_start(self, ctx: TrainContext):
        # set data
        ctx.dataset_path: Path = Path(ctx.dataset_path)
        if self.backend_task_name in ["detect", "segment"]:
            if ctx.dataset_path.suffix not in [".yml", ".yaml"]:
                yaml_files = list(ctx.dataset_path.glob("*.yaml")) + list(
                    ctx.dataset_path.glob("*.yml")
                )
                if len(yaml_files) != 1:
                    raise FileNotFoundError(
                        f"Ambiguous data file. Detected files: {yaml_files}"
                    )
                ctx.dataset_path = Path(yaml_files[0]).absolute()
            else:
                ctx.dataset_path = ctx.dataset_path.absolute()
        elif self.backend_task_name == "classify":
            if not ctx.dataset_path.is_dir():
                raise ValueError(
                    f"Classification dataset should be directory. Not {ctx.dataset_path}"
                )
            ctx.dataset_path = ctx.dataset_path.absolute()
        ctx.dataset_path = str(ctx.dataset_path)

        # pretrained model
        ctx.pretrained_model = (
            ctx.pretrained_model
            if ctx.pretrained_model
            else self.model_type
            + self.model_size
            + self.TASK_SUFFIX[self.backend_task_name]
            + ".pt"
        )

    def training(self, ctx: TrainContext, callback: TrainCallback):

        model = YOLO(ctx.pretrained_model, task=self.backend_task_name)
        model.train(
            data=ctx.dataset_path,
            epochs=ctx.epochs,
            batch=ctx.batch_size,
            imgsz=ctx.image_size,
            rect=ctx.letter_box,
            device=ctx.device,
            workers=ctx.workers,
            seed=ctx.seed,
            verbose=ctx.verbose,
            project=self.hub_dir,
            name=self.ARTIFACT_DIR,
        )
        del model

    def on_train_end(self, ctx: TrainContext):
        io.copy_file(
            self.artifact_dir / "weights" / "best.pt",
            self.best_ckpt_file,
            create_directory=True,
        )
        io.copy_file(
            self.artifact_dir / "weights" / "last.pt",
            self.last_ckpt_file,
            create_directory=True,
        )
        io.save_json(self.get_metrics(), self.metric_file)

    # Inference Hook
    def get_model(
        self, image_size: Union[int, list] = None, parser: ResultParser = None
    ):
        self.check_train_sanity()

        if image_size is None:
            train_config = io.load_yaml(self.train_config_file)
            image_size = train_config.get("image_size")

        # get adapt functions
        preprocess = self.get_preprocess(self.task)
        postprocess = self.get_postprocess(self.task, image_size=image_size)

        # get model
        model = ModelWrapper(
            model=YOLO(self.best_ckpt_file).model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
            parser=parser if parser else None,
        )

        return model

    def on_inference_end(self, ctx: InferenceContext):
        pass

    # Evaluate Hook
    def evaluating(self):
        raise NotImplementedError
