import threading
import warnings
from pathlib import Path
from typing import Union

import tqdm
from torch import nn
from waffle_utils.file import io
from waffle_utils.utils import type_validator

from waffle_dough.type.task_type import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub.model.result_parser import get_parser
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import EvaluateConfig, TrainConfig
from waffle_hub.schema.result import EvaluateResult
from waffle_hub.utils.callback import EvaluateCallback
from waffle_hub.utils.data import get_dataset_class
from waffle_hub.utils.memory import device_context

from .evaluate import evaluate_function


class Evaluator:
    """
    Evaluation manager class
    """

    # evaluate results file name
    EVALUATE_FILE = "evaluate.json"

    def __init__(
        self,
        root_dir: Path,
        model: Union[ModelWrapper, nn.Module],
        task: Union[str, TaskType],
        train_config: TrainConfig = None,
    ):
        self.root_dir = Path(root_dir)
        self.model = model
        self.task = task
        self.train_config = train_config

    # properties
    @property
    def task(self) -> str:
        """Task Name"""
        return self.__task

    @task.setter
    def task(self, v):
        if v not in list(TaskType):
            raise ValueError(f"Invalid task type: {v}" f"Available task types: {list(TaskType)}")
        self.__task = str(v.value) if isinstance(v, TaskType) else str(v).lower()

    @property
    def evaluate_file(self) -> Path:
        """Evaluate File"""
        return self.root_dir / self.EVALUATE_FILE

    @classmethod
    def get_evaluate_result(cls, root_dir: Union[str, Path]) -> list[dict]:
        """Get evaluate result from evaluate file.

        Args:
            root_dir (Union[str, Path]): root directory of evaluate file

        Example:
            >>> Evaluator.get_evaluate_result()
            [
                {
                    "tag": "mAP",
                    "value": 0.5,
                },
            ]

        Returns:
            list[dict]: evaluate result
        """
        evluate_file_path = Path(root_dir) / cls.EVALUATE_FILE
        if not evluate_file_path.exists():
            warnings.warn(f"Evaluate file {evluate_file_path} is not exist. Evaluate First.")
            return []
        return io.load_json(evluate_file_path)

    # methods
    def evaluate(
        self,
        dataset: Union[Dataset, str, Path],
        dataset_root_dir: str = None,
        set_name: str = "test",
        batch_size: int = 4,
        image_size: Union[int, list[int]] = None,
        letter_box: bool = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        half: bool = False,
        workers: int = 2,
        device: str = "0",
        draw: bool = False,
        hold: bool = True,
    ) -> EvaluateResult:
        """Start Evaluate

        Args:
            dataset (Union[Dataset, str]): Waffle Dataset object or path or name.
            dataset_root_dir (str, optional): Waffle Dataset root directory. Defaults to None.
            set_name (str, optional): Waffle Dataset evalutation set name. Defaults to "test".
            batch_size (int, optional): batch size. Defaults to 4.
            image_size (Union[int, list[int]], optional): image size. Defaults to 224.
            letter_box (bool, optional): letter box. Defaults to True.
            confidence_threshold (float, optional): confidence threshold. Not required in classification. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Not required in classification. Defaults to 0.5.
            half (bool, optional): half. Defaults to False.
            workers (int, optional): workers. Defaults to 2.
            device (str, optional): device. Defaults to "0".
            draw (bool, optional): draw. Defaults to False.
            hold (bool, optional): hold. Defaults to True.

        Examples:
            >>> evaluator = Evaluator(...)
            >>> evaluate_result = evaluator.evaluate(
                    dataset=detection_dataset,
                    batch_size=4,
                    image_size=640,
                    letterbox=False,
                    confidence_threshold=0.25,
                    iou_threshold=0.5,
                    workers=4,
                    device="0",
                )
            # or you can use train option by passing None
            >>> evaluator = Evaluator(... , train_config=train_config)
            >>> evaluate_result = evaluator.evaluate(
                    ...
                    image_size=None,  # use train option
                    letterbox=None,  # use train option
                    ...
                )
            >>> evaluate_result.metrics
            [{"tag": "mAP", "value": 0.1}, ...]

        Returns:
            EvaluateResult: evaluate result
        """

        @device_context("cpu" if device == "cpu" else device)
        def inner(dataset: Dataset, callback: EvaluateCallback, result: EvaluateResult):
            try:
                self.before_evaluate(dataset)
                self.on_evaluate_start()
                self.evaluating(callback, dataset)
                self.on_evaluate_end()
                self.after_evaluate(result)
                callback.force_finish()
            except Exception as e:
                if self.evaluate_file.exists():
                    io.remove_file(self.evaluate_file)
                callback.force_finish()
                callback.set_failed()
                raise e

        if "," in device:
            warnings.warn("multi-gpu is not supported in evaluation. use first gpu only.")
            device = device.split(",")[0]

        if isinstance(dataset, (str, Path)):
            dataset = self._load_dataset(dataset, dataset_root_dir)

        self._check_dataset(dataset)

        # overwrite training config
        if image_size is None:
            if self.train_config is not None:
                image_size = self.train_config.image_size
            else:
                image_size = 224  # default image size
        if letter_box is None:
            if self.train_config is not None:
                letter_box = self.train_config.letter_box
            else:
                letter_box = True  # default letter box

        self.eval_cfg = EvaluateConfig(
            dataset_name=dataset.name,
            set_name=set_name,
            batch_size=batch_size,
            image_size=image_size if isinstance(image_size, list) else [image_size, image_size],
            letter_box=letter_box,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half=half,
            workers=workers,
            device="cpu" if device == "cpu" else f"cuda:{device}",
            draw=draw,
            dataset_root_dir=dataset.root_dir,
        )

        callback = EvaluateCallback(100)  # dummy step
        result = EvaluateResult()
        result.callback = callback

        if hold:
            inner(dataset, callback, result)
        else:
            thread = threading.Thread(target=inner, args=(dataset, callback, result), daemon=True)
            callback.register_thread(thread)
            callback.start()

        return result

    def _load_dataset(self, dataset: Union[str, Path], dataset_root_dir: str) -> Dataset:
        if Path(dataset).exists():
            dataset = Path(dataset)
            dataset = Dataset.load(name=dataset.parts[-1], root_dir=dataset.parents[0].absolute())
        elif dataset in Dataset.get_dataset_list(dataset_root_dir):
            dataset = Dataset.load(name=dataset, root_dir=dataset_root_dir)
        else:
            raise FileNotFoundError(f"Dataset {dataset} is not exist.")
        return dataset

    def _check_dataset(self, dataset: Dataset):
        # check task match
        if dataset.task.lower() != self.task.lower():
            raise ValueError(
                f"Dataset task is not matched with hub task. Dataset task: {dataset.task}, Hub task: {self.task}"
            )

    # evalutation hook
    def before_evaluate(self, dataset: Dataset):
        if len(dataset.get_split_ids()[2]) == 0:
            self.eval_cfg.set_name = "val"
            # logger.warning("test set is not exist. use val set instead.")

    def on_evaluate_start(self):
        pass

    def evaluating(self, callback: EvaluateCallback, dataset: Dataset) -> str:
        device = self.eval_cfg.device

        model = self.model.to(device)

        dataloader = get_dataset_class("dataset")(
            dataset,
            self.eval_cfg.image_size,
            letter_box=self.eval_cfg.letter_box,
            set_name=self.eval_cfg.set_name,
        ).get_dataloader(self.eval_cfg.batch_size, self.eval_cfg.workers)

        result_parser = get_parser(self.task)(
            **self.eval_cfg.to_dict(), categories=dataset.get_categories()
        )

        callback._total_steps = len(dataloader) + 1

        preds = []
        labels = []
        for i, (images, image_infos, annotations) in tqdm.tqdm(
            enumerate(dataloader, start=1), total=len(dataloader)
        ):
            result_batch = model(images.to(device))
            result_batch = result_parser(result_batch, image_infos)

            preds.extend(result_batch)
            labels.extend(annotations)

            callback.update(i)

        metrics = evaluate_function(preds, labels, self.task, len(dataset.get_categories()))

        result_metrics = []
        for tag, value in metrics.to_dict().items():
            if isinstance(value, list):
                values = [
                    {
                        "class_name": cat,
                        "value": cat_value,
                    }
                    for cat, cat_value in zip(dataset.get_category_names(), value)
                ]
            else:
                values = value
            result_metrics.append({"tag": tag, "value": values})
        io.save_json(result_metrics, self.evaluate_file)

    def on_evaluate_end(self):
        pass

    def after_evaluate(self, result: EvaluateResult):
        result.eval_metrics = self.get_evaluate_result(self.root_dir)
