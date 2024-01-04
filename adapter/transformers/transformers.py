import os
import warnings
from pathlib import Path
from typing import Callable, Union

import torch
import transformers
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)
from transformers.utils import ModelOutput
from waffle_utils.file import io

from datasets import load_from_disk
from waffle_hub.dataset import Dataset
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.hub.train.adapter.base_manager import BaseManager
from waffle_hub.hub.train.adapter.transformers.train_input_helper import (
    ClassifierInputHelper,
    ObjectDetectionInputHelper,
)
from waffle_hub.hub.train.adapter.transformers.trainer_callback import CustomCallback
from waffle_hub.schema.fields.category import Category
from waffle_hub.type import BackendType, TaskType

from .config import DEFAULT_PARAMS, MODEL_TYPES


class TransformersManager(BaseManager):
    """
    Transformer Training Manager
    """

    # abstract property
    BACKEND_NAME = str(BackendType.TRANSFORMERS.value)
    VERSION = "4.34.1"
    MODEL_TYPES = MODEL_TYPES
    MULTI_GPU_TRAIN = False
    DEFAULT_PARAMS = DEFAULT_PARAMS
    DEFAULT_ADVANCE_PARAMS = {}

    # Override
    LAST_CKPT_FILE = "last_ckpt"
    BEST_CKPT_FILE = "best_ckpt"

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

        if self.VERSION is not None and transformers.__version__ != self.VERSION:
            warnings.warn(
                f"You've loaded the Hub created with transformers=={self.VERSION}, \n"
                + f"but the installed version is {transformers.__version__}."
            )

    # override
    @property
    def last_ckpt_file(self) -> Path:
        return self.weights_dir / TransformersManager.LAST_CKPT_FILE

    @property
    def best_ckpt_file(self) -> Path:
        return self.weights_dir / TransformersManager.BEST_CKPT_FILE

    # Model
    def get_model(self) -> ModelWrapper:
        self.check_train_sanity()

        # get adapt functions
        preprocess = self._get_preprocess()
        postprocess = self._get_postprocess()

        # get model
        if self.task == TaskType.OBJECT_DETECTION:
            model = AutoModelForObjectDetection.from_pretrained(
                str(self.best_ckpt_file),
            )
        elif self.task == TaskType.CLASSIFICATION:
            model = AutoModelForImageClassification.from_pretrained(
                str(self.best_ckpt_file),
            )
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented.")

        model = ModelWrapper(
            model=model.eval(),
            preprocess=preprocess,
            postprocess=postprocess,
        )

        return model

    def _get_preprocess(self, pretrained_model: str = None) -> Callable:
        if pretrained_model is None:
            pretrained_model = self.best_ckpt_file
        image_processer = AutoImageProcessor.from_pretrained(pretrained_model)

        normalize = T.Normalize(image_processer.image_mean, image_processer.image_std, inplace=True)

        def preprocess(x, *args, **kwargs):
            return normalize(x)

        return preprocess

    def _get_postprocess(self: str, pretrained_model: str = None) -> Callable:
        if pretrained_model is None:
            pretrained_model = self.best_ckpt_file
        image_processer = AutoImageProcessor.from_pretrained(pretrained_model)

        if self.task == TaskType.CLASSIFICATION:

            def inner(x: ModelOutput, *args, **kwargs) -> torch.Tensor:
                return [x.logits]

        elif self.task == TaskType.OBJECT_DETECTION:
            post_process = image_processer.post_process_object_detection

            def inner(x: ModelOutput, *args, **kwargs) -> torch.Tensor:

                x = post_process(x, threshold=-1)

                xyxy = list(map(lambda x: x["boxes"], x))
                confidences = list(map(lambda x: x["scores"], x))
                category_ids = list(map(lambda x: x["labels"], x))

                return xyxy, confidences, category_ids

        else:
            raise NotImplementedError(f"Task {self.task} is not implemented.")

        return inner

    # Trainer
    # Train hook
    def on_train_start(self):
        # overwrite train config with default config
        self.train_cfg.pretrained_model = self.MODEL_TYPES[self.task][self.model_type][
            self.model_size
        ]

        dataset = load_from_disk(self.train_cfg.dataset_path)

        if self.task == TaskType.CLASSIFICATION:
            helper = ClassifierInputHelper(
                self.train_cfg.pretrained_model, self.train_cfg.image_size
            )
            self.train_cfg.train_input = helper.get_train_input()
            categories = dataset["train"].features["label"].names
            id2label = {index: x for index, x in enumerate(categories, start=0)}
            self.train_cfg.train_input.model = AutoModelForImageClassification.from_pretrained(
                self.train_cfg.pretrained_model,
                num_labels=len(id2label),
                ignore_mismatched_sizes=True,
            )

        elif self.task == TaskType.OBJECT_DETECTION:
            helper = ObjectDetectionInputHelper(
                self.train_cfg.pretrained_model, self.train_cfg.image_size
            )
            self.train_cfg.train_input = helper.get_train_input()
            categories = dataset["train"].features["objects"].feature["category"].names
            id2label = {index: x for index, x in enumerate(categories, start=0)}
            label2id = {x: index for index, x in enumerate(categories, start=0)}
            self.train_cfg.train_input.model = AutoModelForObjectDetection.from_pretrained(
                self.train_cfg.pretrained_model,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            raise NotImplementedError

        if self.train_cfg.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.train_cfg.train_input.model = self.train_cfg.train_input.model.to("cpu")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.train_cfg.device

        transforms = helper.get_transforms()
        dataset["train"] = dataset["train"].with_transform(transforms)
        dataset["val"] = dataset["val"].with_transform(transforms)
        self.train_cfg.train_input.dataset = dataset

        self.train_cfg.train_input.training_args = TrainingArguments(
            output_dir=str(self.artifacts_dir),
            per_device_train_batch_size=self.train_cfg.batch_size,
            num_train_epochs=self.train_cfg.epochs,
            remove_unused_columns=False,
            push_to_hub=False,
            logging_strategy="epoch" if self.train_cfg.verbose else "no",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.train_cfg.learning_rate,
            dataloader_num_workers=self.train_cfg.workers,
            seed=self.train_cfg.seed,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            load_best_model_at_end=False,
            use_cpu=(self.train_cfg.device == "cpu"),
        )

    def training(self):
        trainer = Trainer(
            model=self.train_cfg.train_input.model,
            args=self.train_cfg.train_input.training_args,
            data_collator=self.train_cfg.train_input.collator,
            train_dataset=self.train_cfg.train_input.dataset["train"],
            eval_dataset=self.train_cfg.train_input.dataset["val"],
            tokenizer=self.train_cfg.train_input.image_processor,
            compute_metrics=self.train_cfg.train_input.compute_metrics,
        )
        trainer.add_callback(CustomCallback(trainer, self.metric_file))
        trainer.train()
        trainer.save_model(str(self.artifacts_dir / "weights" / self.LAST_CKPT_FILE))
        trainer._load_best_model()
        trainer.save_model(str(self.artifacts_dir / "weights" / self.BEST_CKPT_FILE))

    def on_train_end(self):
        io.copy_files_to_directory(
            self.artifacts_dir / "weights" / self.BEST_CKPT_FILE,
            self.best_ckpt_file,
            create_directory=True,
        )
        io.copy_files_to_directory(
            self.artifacts_dir / "weights" / self.LAST_CKPT_FILE,
            self.last_ckpt_file,
            create_directory=True,
        )
        io.save_json(self.get_metrics(), self.metric_file)

    def get_metrics(self) -> list[list[dict]]:
        return io.load_json(self.metric_file) if self.metric_file.exists() else []
