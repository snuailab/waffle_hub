import warnings
from pathlib import Path
from typing import Any, Union

import tqdm
from torch import nn
from waffle_utils.file import io

from waffle_dough.type.task_type import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub.model.result_parser import get_parser
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import EvaluateConfig, TrainConfig
from waffle_hub.schema.result import EvaluateResult
from waffle_hub.schema.running_status import EvaluatingStatus
from waffle_hub.utils import hook
from waffle_hub.utils.data import get_dataset_class
from waffle_hub.utils.memory import device_context

from .evaluate import evaluate_function


class BaseEvaluateHooks(hook.BaseHooks):
    def __init__(self):
        pass

    # hooks
    def before_evaluate(self, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult):
        pass

    def on_evaluate_start(
        self, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        pass

    def on_evaluate_loop_start(
        self, total_step: int, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        pass

    def on_evaluate_step_start(
        self, step: int, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        pass

    def on_evaluate_step_end(
        self, step: int, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        pass

    def on_evaluate_loop_end(
        self, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        pass

    def on_evaluate_end(self, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult):
        pass

    def after_evaluate(self, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult):
        pass

    def on_exception_stopped(
        self, e: Exception, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        pass

    def on_exception_failed(
        self, e: Exception, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        pass


class DefaultEvaluateHooks(BaseEvaluateHooks):
    def __init__(self, evaluate_file_path: Path):
        super().__init__()
        self.evaluate_file_path = evaluate_file_path

    # evalutation hooks
    def before_evalutae(self, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult):
        status.set_running()

    def on_evaluate_start(
        self, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        pass

    def on_evaluate_loop_start(
        self, total_step: int, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        status.set_total_step(total_step)

    def on_evaluate_step_start(
        self, step: int, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        pass

    def on_evaluate_step_end(
        self, step: int, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        status.set_current_step(step)

    def on_evaluate_loop_end(
        self, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        pass

    def on_evaluate_end(
        self,
        result_metrics: list[dict],
        cfg: EvaluateConfig,
        status: EvaluatingStatus,
        result: EvaluateResult,
    ):
        result.eval_metrics = result_metrics
        io.save_json(result_metrics, self.evaluate_file_path)

    def after_evaluate(self, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult):
        status.set_success()

    def on_exception_stopped(
        self, e: Exception, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        status.set_stopped(e)

    def on_exception_failed(
        self, e: Exception, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        status.set_failed(e)


class Evaluator:
    """
    Evaluation manager class
    """

    # evaluate results file name
    EVALUATE_FILE = "evaluate.json"

    def __init__(
        self,
        root_dir: Path,
        model: ModelWrapper,
        task: Union[str, TaskType],
    ):
        self.root_dir = Path(root_dir)
        self.model = model
        self.task = task

        # hooks
        hook.initalize_hooks(self, DefaultEvaluateHooks(evaluate_file_path=self.evaluate_file))

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
    @device_context
    def evaluate(
        self,
        dataset: Union[Dataset, str],
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
    ) -> EvaluateResult:
        """Start Evaluate

        Args:
            dataset (Union[Dataset, str]): Waffle Dataset object or path or name.
            dataset_root_dir (str, optional): Waffle Dataset root directory. Defaults to None.
            set_name (str, optional): Eval set name. Defaults to "test".
            batch_size (int, optional): batch size. Defaults to 4.
            image_size (Union[int, list[int]], optional): image size. Defaults to None.
            letter_box (bool, optional): letter box. Defaults to None.
            confidence_threshold (float, optional): confidence threshold. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Defaults to 0.5.
            half (bool, optional): half. Defaults to False.
            workers (int, optional): workers. Defaults to 2.
            device (str, optional): device. Defaults to "0".
            draw (bool, optional): draw. Defaults to False.

        Raises:
            FileNotFoundError: if can not detect appropriate dataset.

        Examples:
            >>> evaluate_result = hub.evaluate(
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
            >>> evaluate_result = hub.evaluate(
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

        try:
            status = EvaluatingStatus(root_dir=self.root_dir)
            if "," in device:
                warnings.warn("multi-gpu is not supported in evaluation. use first gpu only.")
                device = device.split(",")[0]

            if isinstance(dataset, (str, Path)):
                dataset = self._load_dataset(dataset, dataset_root_dir)

            self.model.is_valid_dataset(dataset)

            # config setting
            # overwrite training config or default
            if image_size is None:
                if self.model.train_config is not None:
                    image_size = self.model.train_config.image_size
                else:
                    image_size = 224  # default image size
            if letter_box is None:
                if self.model.train_config is not None:
                    letter_box = self.model.train_config.letter_box
                else:
                    letter_box = True  # default letter box

            if set_name == "test" and len(dataset.get_split_ids()[2]) == 0:
                set_name = "val"
                # logger.warning("test set is not exist. use val set instead.")

            cfg = EvaluateConfig(
                dataset_name=dataset.name,
                dataset_root_dir=dataset.root_dir,
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
            )

            result = EvaluateResult()

            # evaluate run
            hook.run_hooks(self, "before_evaluate", cfg, status, result)
            self._evaluate(dataset, cfg, status, result)
            hook.run_hooks(self, "after_evaluate", cfg, status, result)

        except (KeyboardInterrupt, SystemExit) as e:
            hook.run_hooks(self, "on_exception_stopped", e, cfg, status, result)
            raise e
        except Exception as e:
            hook.run_hooks(self, "on_exception_failed", e, cfg, status, result)
            if self.evaluate_file.exists():
                io.remove_file(self.evaluate_file)
            raise e

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

    def _evaluate(
        self, dataset: Dataset, cfg: EvaluateConfig, status: EvaluatingStatus, result: EvaluateResult
    ):
        hook.run_hooks(self, "on_evaluate_start", cfg, status, result)

        device = cfg.device
        model = self.model.to(device)
        dataloader = get_dataset_class("dataset")(
            dataset,
            cfg.image_size,
            letter_box=cfg.letter_box,
            set_name=cfg.set_name,
        ).get_dataloader(cfg.batch_size, cfg.workers)

        result_parser = get_parser(self.task)(**cfg.to_dict(), categories=dataset.get_categories())

        hook.run_hooks(self, "on_evaluate_loop_start", len(dataloader) + 1, cfg, status, result)

        preds = []
        labels = []
        for i, (images, image_infos, annotations) in tqdm.tqdm(
            enumerate(dataloader, start=1), total=len(dataloader)
        ):
            hook.run_hooks(self, "on_evaluate_step_start", i, cfg, status, result)
            result_batch = model(images.to(device))
            result_batch = result_parser(result_batch, image_infos)

            preds.extend(result_batch)
            labels.extend(annotations)

            hook.run_hooks(self, "on_evaluate_step_end", i, cfg, status, result)

        hook.run_hooks(self, "on_evaluate_loop_end", cfg, status, result)

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

        hook.run_hooks(self, "on_evaluate_end", result_metrics, cfg, status, result)
