import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Union

import torch
import ultralytics
from torch import nn
from torchvision import transforms as T
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG as YOLO_DEFAULT_ADVANCE_PARAMS
from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_dough.type.task_type import TaskType
from waffle_hub import EXPORT_MAP
from waffle_hub.dataset import Dataset
from waffle_hub.hub.manager.base_manager import BaseManager
from waffle_hub.hub.model.base_model import Model
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.hub.train.base_trainer import Trainer
from waffle_hub.hub.train.trainer import BaseTrainHooks
from waffle_hub.schema.configs import ModelConfig, TrainConfig
from waffle_hub.schema.fields.category import Category
from waffle_hub.schema.result import TrainResult
from waffle_hub.schema.running_status import TrainingStatus
from waffle_hub.type.backend_type import BackendType
from waffle_hub.utils.process import run_python_file

from .config import DEFAULT_PARAMS, MODEL_TYPES, TASK_MAP


class UltralyticsManager(BaseManager):
    """
    Ultralytics Manager (Train, Model)
    """

    # abstract property

    BACKEND_NAME = str(BackendType.ULTRALYTICS.value)
    BACKEND_VERSION = "8.0.201"
    MULTI_GPU_TRAIN = True
    MODEL_TYPES = MODEL_TYPES
    DEFAULT_PARAMS = DEFAULT_PARAMS
    DEFAULT_ADVANCE_PARAMS = dict(YOLO_DEFAULT_ADVANCE_PARAMS)

    TASK_MAP = TASK_MAP

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

        if self.VERSION is not None and ultralytics.__version__ != self.VERSION:
            warnings.warn(
                f"You've loaded the Hub created with ultralytics=={self.VERSION}, \n"
                + f"but the installed version is {ultralytics.__version__}."
            )
        self.backend_task_name = self.TASK_MAP[self.task]

    def get_default_advance_train_params(
        cls, task: str = None, model_type: str = None, model_size: str = None
    ) -> dict:
        return cls.DEFAULT_ADVANCE_PARAMS

    # Model
    def get_model(self) -> ModelWrapper:
        self.check_train_sanity()

        # get model
        model = YOLO(self.best_ckpt_file).model

        # get adapt functions
        names: list[str] = list(map(lambda x: x["name"], self.categories))
        yolo_names: dict[int, str] = model.names
        if len(names) != len(yolo_names):
            raise ValueError(
                f"Number of categories is not matched. hub: {len(names)} != ultralytics: {len(yolo_names)}"
            )

        id_mapper: list[int] = [i for i in range(len(yolo_names))]

        yolo_names_inv = {v: k for k, v in yolo_names.items()}
        for i, name in enumerate(names):
            id_mapper[i] = yolo_names_inv[name]

        # get adapt functions
        preprocess = self._get_preprocess()
        postprocess = self._get_postprocess(id_mapper=id_mapper)

        # return model wrapper
        return ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
        )

    def _get_preprocess(self, *args, **kwargs):

        if self.task == TaskType.CLASSIFICATION:
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x, *args, **kwargs):
                return normalize(x)

        elif self.task == TaskType.OBJECT_DETECTION:
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x, *args, **kwargs):
                return normalize(x)

        elif self.task == TaskType.INSTANCE_SEGMENTATION:
            normalize = T.Normalize([0, 0, 0], [1, 1, 1], inplace=True)

            def preprocess(x, *args, **kwargs):
                return normalize(x)

        else:
            raise NotImplementedError(f"Task {self.task} is not implemented.")

        return preprocess

    def _get_postprocess(self, *args, **kwargs):

        id_mapper: list[int] = kwargs.get("id_mapper", [i for i in range(len(self.categories))])

        if self.task == TaskType.CLASSIFICATION:

            def inner(x: torch.Tensor, *args, **kwargs):
                return [x[:, id_mapper]]

        elif self.task == TaskType.OBJECT_DETECTION:

            def inner(x: torch.Tensor, image_size: tuple[int, int], *args, **kwargs):
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
                probs = x[:, :, 4:][:, :, id_mapper]
                confidences, class_ids = torch.max(probs, dim=-1)

                return xyxy, confidences, class_ids

        elif self.task == TaskType.INSTANCE_SEGMENTATION:
            num_category = len(self.categories)

            def inner(x: torch.Tensor, image_size: tuple[int, int], *args, **kwargs):
                preds = x[0]  # x[0]: prediction, x[1]: TODO: what is this...?
                preds = preds.transpose(1, 2)

                probs = preds[:, :, 4 : 4 + num_category]
                confidences, class_ids = torch.max(probs, dim=-1)
                _, indicies = torch.topk(
                    confidences, k=min(100, confidences.shape[-1]), dim=-1, largest=True
                )  # TODO: make k configurable

                preds = torch.gather(preds, 1, indicies.unsqueeze(-1).repeat(1, 1, preds.shape[-1]))
                confidences = torch.gather(confidences, 1, indicies)
                class_ids = torch.gather(class_ids, 1, indicies)

                cxcywh = preds[:, :, :4]
                cx, cy, w, h = torch.unbind(cxcywh, dim=-1)
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                xyxy[:, :, ::2] /= image_size[0]
                xyxy[:, :, 1::2] /= image_size[1]

                # [batch, mask_dim, mask_height, mask_width] -> [batch, num_mask, mask_height * mask_width]
                protos = x[1][-1] if len(x[1]) == 3 else x[1]
                mask_size = (protos.shape[-2], protos.shape[-1])
                protos = protos.view(
                    protos.shape[0], -1, protos.shape[-2] * protos.shape[-1]
                ).float()

                masks = preds[:, :, 4 + num_category :]
                masks = torch.bmm(masks, protos).sigmoid().view(masks.shape[0], -1, *mask_size)

                return xyxy, confidences, class_ids, masks

        else:
            raise NotImplementedError(f"Task {self.task} is not implemented.")

        return inner

    # Trainer abstract method
    @abstractmethod
    def _init_trainer(self):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError


class UltralyticsTrainHooks(BaseTrainHooks):
    # hooks
    def before_train(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass

    def on_train_start(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        # set data
        cfg.dataset_path: Path = Path(cfg.dataset_path)
        if self.backend_task_name in ["detect", "segment"]:
            if cfg.dataset_path.suffix not in [".yml", ".yaml"]:
                yaml_files = list(cfg.dataset_path.glob("*.yaml")) + list(
                    cfg.dataset_path.glob("*.yml")
                )
                if len(yaml_files) != 1:
                    raise FileNotFoundError(f"Ambiguous data file. Detected files: {yaml_files}")
                cfg.dataset_path = Path(yaml_files[0]).absolute()
            else:
                cfg.dataset_path = cfg.dataset_path.absolute()
        elif self.backend_task_name == "classify":
            if not cfg.dataset_path.is_dir():
                raise ValueError(
                    f"Classification dataset should be directory. Not {cfg.dataset_path}"
                )
            cfg.dataset_path = cfg.dataset_path.absolute()
        cfg.dataset_path = str(cfg.dataset_path)

        # pretrained model
        cfg.pretrained_model = (
            cfg.pretrained_model
            if cfg.pretrained_model
            else self.MODEL_TYPES[self.task][self.model_type][self.model_size]
        )

        # other
        if self.task in [TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]:
            cfg.letter_box = True  # TODO: hard coding
            # logger.warning(
            #     "letter_box False is not supported for Object Detection and Segmentation."
            # )

    def training(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        params = {
            "data": str(cfg.dataset_path).replace("\\", "/"),
            "epochs": cfg.epochs,
            "batch": cfg.batch_size,
            "imgsz": cfg.image_size,
            "lr0": cfg.learning_rate,
            "lrf": cfg.learning_rate,
            "rect": False,  # TODO: hard coding for mosaic
            "device": cfg.device,
            "workers": cfg.workers,
            "seed": cfg.seed,
            "verbose": cfg.verbose,
            "project": str(self.root_dir),
            "name": str(self.ARTIFACTS_DIR),
        }
        params.update(cfg.advance_params)

        code = f"""if __name__ == "__main__":
        from ultralytics import YOLO
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        try:
            model = YOLO("{cfg.pretrained_model}", task="{self.backend_task_name}")
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

    def on_train_end(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
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

    def after_train(self, cfg: TrainConfig, status: TrainingStatus, result: TrainResult):
        pass
