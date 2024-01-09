import warnings
from pathlib import Path
from typing import Union

import torch
import ultralytics
from torchvision import transforms as T
from ultralytics.utils import DEFAULT_CFG as YOLO_DEFAULT_ADVANCE_PARAMS
from waffle_utils.callback import BaseCallback

from waffle_hub.hub.manager.adapter.ultralytics.ultralytics_callback import (
    UltralyticsTrainCallback,
)
from waffle_hub.hub.manager.base_manager import BaseManager
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.fields.category import Category
from waffle_hub.schema.state import TrainState
from waffle_hub.type import BackendType, TaskType

from .config import DEFAULT_PARAMS, MODEL_TYPES, TASK_MAP


class UltralyticsManager(BaseManager):
    """
    Ultralytics Manager
    """

    BACKEND_NAME = str(BackendType.ULTRALYTICS.value)
    BACKEND_VERSION = "8.0.227"
    MODEL_TYPES = MODEL_TYPES
    MULTI_GPU_TRAIN = True
    DEFAULT_PARAMS = DEFAULT_PARAMS
    DEFAULT_ADVANCE_PARAMS = dict(YOLO_DEFAULT_ADVANCE_PARAMS)

    TASK_MAP = TASK_MAP

    def __init__(
        self,
        root_dir: Path,
        name: str,
        task: Union[str, TaskType] = None,
        model_type: str = None,
        model_size: str = None,
        categories: list[Union[str, int, float, dict, Category]] = None,
        callbacks: list[BaseCallback] = None,
        load: bool = False,
        train_state: TrainState = None,
    ):
        ultralytics_callbacks = [UltralyticsTrainCallback()]
        if callbacks is not None:
            ultralytics_callbacks.extend(callbacks)

        super().__init__(
            root_dir=root_dir,
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
            callbacks=ultralytics_callbacks,
            load=load,
            train_state=train_state,
        )

        if self.BACKEND_VERSION is not None and ultralytics.__version__ != self.BACKEND_VERSION:
            warnings.warn(
                f"You've loaded the Hub created with ultralytics=={self.BACKEND_VERSION}, \n"
                + f"but the installed version is {ultralytics.__version__}."
            )

        self.backend_task_name = self.TASK_MAP[self.task]

    # Model
    def get_model(self) -> ModelWrapper:
        self.check_train_sanity()

        # get model
        model = ultralytics.YOLO(self.best_ckpt_file).model

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
            task=self.task,
            categories=self.categories,
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

    # Trainer
    def get_metrics(self) -> list[list[dict]]:
        # read csv file
        # epoch,         train/box_loss,         train/cls_loss,         train/dfl_loss,   metrics/precision(B),      metrics/recall(B),       metrics/mAP50(B),    metrics/mAP50-95(B),           val/box_loss,           val/cls_loss,           val/dfl_loss,                 lr/pg0,                 lr/pg1,                 lr/pg2
        #              0,                 2.0349,                 4.4739,                  1.214,                0.75962,                0.34442,                0.22858,                0.18438,                0.61141,                 2.8409,                0.78766,                 0.0919,                 0.0009,                 0.0009
        #              1,                 1.7328,                 4.0078,                 1.1147,                0.12267,                0.40909,                0.20891,                0.15007,                0.75082,                 2.7157,                0.79723,               0.082862,              0.0018624,              0.0018624
        # and parse it to list[dict]
        # [[{"tag": train/box_loss, "value": 0.0}, {"tag": train/box_loss, "value": 1.7328}, ...], ...]
        # and return it

        csv_path = self.artifacts_dir / "results.csv"

        if not csv_path.exists():
            return []

        with open(csv_path) as f:
            lines = f.readlines()

        header = lines[0].strip().split(",")
        metrics = []
        for line in lines[1:]:
            values = line.strip().split(",")
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

    def get_default_advance_train_params(
        cls, task: str = None, model_type: str = None, model_size: str = None
    ) -> dict:
        return cls.DEFAULT_ADVANCE_PARAMS
