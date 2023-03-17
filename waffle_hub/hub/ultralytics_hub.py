"""
Ultralytics Hub
"""

from waffle_hub import get_installed_backend_version

BACKEND_NAME = "ultralytics"
BACKEND_VERSION = get_installed_backend_version(BACKEND_NAME)

from dataclasses import asdict
from pathlib import Path

import torch
import tqdm
from torchvision import transforms as T
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
from waffle_utils.dataset.fields import Annotation
from waffle_utils.file import io

from waffle_hub.schemas.configs import (
    Classes,
    ClassificationPrediction,
    DetectionPrediction,
    Train,
)
from waffle_hub.utils.image import ImageDataset

from . import BaseHub
from .model import ModelWrapper, get_result_parser


def get_preprocess(task: str, *args, **kwargs):

    if task == "classification":
        normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

        def preprocess(x):
            return normalize(x)

    elif task == "object_detection":
        normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

        def preprocess(x):
            return normalize(x)

    return preprocess


def get_postprocess(task: str, *args, **kwargs):

    if task == "classification":

        def inner(x):
            return x

    elif task == "object_detection":
        image_size = kwargs.get("image_size")
        image_size = (
            image_size
            if isinstance(image_size, list)
            else [image_size, image_size]
        )

        def inner(x):
            x = x[0]
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

    return inner


class UltralyticsHub(BaseHub):

    # Common
    MODEL_TYPES = ["yolov8"]
    MODEL_SIZES = list("nsmlx")  # TODO: generalize

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
        backend: str = None,
        version: str = None,
        task: str = None,
        model_type: str = None,
        model_size: str = None,
        root_dir: str = None,
    ):
        """Create Ultralytics Hub.

        Args:
            name (str): Hub name
            backend (str, optional): Backend name. See waffle_hub.get_backends(). Defaults to None.
            version (str, optional): Version. See waffle_hub.get_installed_backend_version(backend). Defaults to None.
            task (str, optional): Task Name. See UltralyticsHub.TASKS. Defaults to None.
            model_type (str, optional): Model Type. See UltralyticsHub.MODEL_TYPES. Defaults to None.
            model_size (str, optional): Model Size. See UltralyticsHub.MODEL_SIZES. Defaults to None.
            root_dir (str, optional): Root directory of hub repository. Defaults to None.
        """
        super().__init__(
            name=name,
            backend=backend if backend else BACKEND_NAME,
            version=version if version else BACKEND_VERSION,
            task=task,
            model_type=model_type,
            model_size=model_size,
            root_dir=root_dir,
        )

        self.backend_task_name = self.TASK_MAP.get(self.task, None)
        if self.backend_task_name is None:
            raise ValueError(
                f"{self.task} is not supported with {self.backend}"
            )

    def train(
        self,
        dataset_dir: str,
        epochs: int,
        batch_size: int,
        image_size: int,
        pretrained_model: str = None,
        device: str = "0",
        workers: int = 2,
        seed: int = 0,
        verbose: bool = True,
    ) -> str:
        """Start Train

        Args:
            dataset_dir (str): Dataset Directory. Recommend to use result of waffle_utils.dataset.Dataset.export.
            epochs (int): total epochs
            batch_size (int): batch size
            image_size (int): image size
            pretrained_model (str, optional): pretrained model file. Defaults to None.
            device (str, optional): gpu device. Defaults to "0".
            workers (int, optional): num workers. Defaults to 2.
            seed (int, optional): random seed. Defaults to 0.
            verbose (bool, optional): verbose. Defaults to True.

        Raises:
            FileExistsError: if trained artifact exists.
            FileNotFoundError: if can not detect appropriate dataset.
            ValueError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

        Returns:
            str: hub directory
        """
        if self.artifact_dir.exists():
            raise FileExistsError(
                "Train artifacts already exist. Remove artifact to re-train (hub.delete_artifact())."
            )

        # set data
        dataset_dir: Path = Path(dataset_dir)
        if self.backend_task_name in ["detect", "segment"]:
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
        elif self.backend_task_name == "classify":
            if not dataset_dir.is_dir():
                raise ValueError(
                    f"Classification dataset should be directory. Not {dataset_dir}"
                )
            data = dataset_dir.absolute()
        data = str(data)

        # pretrained model
        pretrained_model = (
            pretrained_model
            if pretrained_model
            else self.model_type
            + self.model_size
            + self.TASK_SUFFIX[self.backend_task_name]
            + ".pt"
        )

        # save train config.
        io.save_yaml(
            asdict(
                Train(
                    image_size=image_size,
                    batch_size=batch_size,
                    pretrained_model=pretrained_model,
                    seed=seed,
                )
            ),
            self.train_config_file,
            create_directory=True,
        )

        try:
            model = YOLO(pretrained_model, task=self.backend_task_name)
            model.train(
                data=data,
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                device=device,
                workers=workers,
                seed=seed,
                verbose=verbose,
                project=self.hub_dir,
                name=self.RAW_TRAIN_DIR,
            )

            # save classes config.
            io.save_yaml(
                asdict(Classes(names=model.names)),
                self.classes_config_file,
                create_directory=True,
            )

        except Exception as e:
            if self.artifact_dir.exists():
                io.remove_directory(self.artifact_dir)
            raise e

        # Parse Training Results
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
        io.copy_file(
            self.artifact_dir / "results.csv",
            self.metric_file,
            create_directory=True,
        )

        return str(self.hub_dir)

    def inference(
        self,
        source: str,
        batch_size: int,
        recursive: bool = True,
        image_size: int = None,
        confidence_threshold: float = 0.25,
        iou_thresold: float = 0.5,
        half: bool = False,
        workers: int = 2,
        device: str = "0",
    ) -> str:
        """Start Inference

        Args:
            source (str): dataset source. image file or image directory. TODO: video
            recursive (bool, optional): get images from directory recursively. Defaults to True.
            image_size (int, optional): inference image size. None to load image_size from train_config (recommended).
            conf_thres (float, optional): confidence threshold. Defaults to 0.25.
            iou_thres (float, optional): iou threshold. Defaults to 0.7.
            half (bool, optional): fp16 inference. Defaults to False.
            device (str, optional): gpu device. Defaults to "0".

        Raises:
            FileNotFoundError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

        Returns:
            str: inference result directory
        """
        self.check_train_sanity()

        # overwrite training config
        train_config = io.load_yaml(self.train_config_file)
        image_size = (
            image_size if image_size else train_config.get("image_size")
        )

        # get adapt functions
        preprocess = get_preprocess(self.task)
        postprocess = get_postprocess(self.task, image_size=image_size)
        parser = get_result_parser(
            self.task, confidence_threshold, iou_thresold
        )

        # get images
        image_dataset = ImageDataset(
            source, image_size, letter_box=False
        )  # TODO: add letter box option in train

        # inference
        device = "cpu" if device == "cpu" else f"cuda:{device}"

        model = YOLO(self.best_ckpt_file).model.eval()
        model = ModelWrapper(
            model, preprocess=preprocess, postprocess=postprocess
        )
        model.to(device)

        dl = image_dataset.get_dataloader(batch_size, workers)
        for images, image_infos in tqdm.tqdm(dl, total=len(dl)):
            results = model(images.to(device))
            results = parser(results, image_infos)
            results

    def evaluation(self):
        raise NotImplementedError

    def export(self):
        self.check_train_sanity()

        train_config = Train(**io.load_yaml(self.train_config_file))
        dynamic_batch = 16
        image_size = [train_config.image_size, train_config.image_size]

        model = YOLO(self.best_ckpt_file).model
        model = ModelWrapper(
            model, preprocess=preprocess(), postprocess=postprocess(image_size)
        )

        input_name = ["inputs"]
        output_names = ["bbox", "conf", "class_id"]

        dummy_input = torch.randn(dynamic_batch, 3, *image_size)

        torch.onnx.export(
            model,
            dummy_input,
            str(self.onnx_file),
            input_names=input_name,
            output_names=output_names,
            opset_version=11,
            dynamic_axes={
                name: {0: "batch_size"} for name in input_name + output_names
            },
        )

        return str(self.onnx_file)
