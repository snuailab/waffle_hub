import os
import warnings
from pathlib import Path
from typing import Union

import torch
import ultralytics
from torch import nn
from torchvision import transforms as T
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG as YOLO_DEFAULT_ADVANCE_PARAMS
from waffle_utils.file import io

from waffle_dough.type.task_type import TaskType
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.hub.train.adapter.base_manager import BaseManager
from waffle_hub.schema.fields.category import Category
from waffle_hub.type.backend_type import BackendType
from waffle_hub.utils.process import run_python_file

from .config import DEFAULT_PARAMS, MODEL_TYPES, TASK_MAP


class UltralyticsManager(BaseManager):
    """
    Ultralytics Training Manager
    """

    BACKEND_NAME = str(BackendType.ULTRALYTICS.value)
    VERSION = "8.0.201"
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
    ):
        super().__init__(
            root_dir=root_dir,
            name=name,
            task=task,
            model_type=model_type,
            model_size=model_size,
            categories=categories,
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
        preprocess = self.get_preprocess()
        postprocess = self.get_postprocess(id_mapper=id_mapper)

        # return model wrapper
        return ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
        )

    def get_preprocess(self, *args, **kwargs):

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

    def get_postprocess(self, *args, **kwargs):

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

        csv_path = self.artifact_dir / "results.csv"

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
            "name": str(self.ARTIFACT_DIR),
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
