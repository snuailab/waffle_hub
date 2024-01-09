import warnings
from pathlib import Path
from typing import Union

import tqdm
from waffle_utils.callback import BaseCallback
from waffle_utils.file import io

from waffle_hub import EvaluateStatus
from waffle_hub.dataset import Dataset
from waffle_hub.hub.model.result_parser import get_parser
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import EvaluateConfig
from waffle_hub.schema.result import EvaluateResult
from waffle_hub.schema.state import EvaluateState
from waffle_hub.utils.data import get_dataset_class
from waffle_hub.utils.memory import device_context

from .evaluate import evaluate_function
from .hook import BaseEvaluateHook


class Evaluator(BaseEvaluateHook):
    """
    Evaluation manager class
    """

    # evaluate results file name
    EVALUATE_FILE = "evaluate.json"

    def __init__(
        self,
        root_dir: Path,
        model: ModelWrapper,
        callbacks: list[BaseCallback] = None,
    ):
        super().__init__(callbacks)
        self.root_dir = Path(root_dir)
        self.model = model

        # state
        self.state = EvaluateState(status=EvaluateStatus.INIT)
        # result
        self.result = EvaluateResult()

    @property
    def evaluate_file(self) -> Path:
        """Evaluate File"""
        return self.root_dir / self.EVALUATE_FILE

    @classmethod
    def get_evaluate_result(cls, root_dir: Union[str, Path]) -> list[dict]:
        """Get evaluate result from evaluate file.

        Args:
            root_dir (Union[str, Path]): root directory of evaluate file

        Examples:
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
        dataset: Union[Dataset, Path, str],
        dataset_root_dir: Union[Path, str] = None,
        set_name: str = "test",
        batch_size: int = 4,
        image_size: Union[int, list[int]] = [640, 640],
        letter_box: bool = True,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        half: bool = False,
        workers: int = 2,
        device: str = "0",
    ) -> EvaluateResult:
        """Start Evaluate

        Args:
            dataset (Union[Dataset, Path, str]): Waffle Dataset object or path or name.
            dataset_root_dir (Union[Path, str], optional): Waffle Dataset root directory. Defaults to None.
            set_name (str, optional): Eval set name. Defaults to "test".
            batch_size (int, optional): batch size. Defaults to 4.
            image_size (Union[int, list[int]], optional): image size. Defaults to [640, 640].
            letter_box (bool, optional): letter box. Defaults to True.
            confidence_threshold (float, optional): confidence threshold. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Defaults to 0.5.
            half (bool, optional): half. Defaults to False.
            workers (int, optional): workers. Defaults to 2.
            device (str, optional): device. Defaults to "0".

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
            self.run_default_hook("setup")
            self.run_callback_hooks("setup", self)

            # load dataset
            if isinstance(dataset, (str, Path)):
                dataset = self._load_dataset(dataset, dataset_root_dir)

            self.is_valid_dataset(dataset)

            # config setting
            # check device
            if "," in device:
                warnings.warn("multi-gpu is not supported in evaluation. use first gpu only.")
                device = device.split(",")[0]

            if set_name == "test" and len(dataset.get_split_ids()[2]) == 0:
                set_name = "val"
                # logger.warning("test set is not exist. use val set instead.")

            self.cfg = EvaluateConfig(
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
            )

            self.run_default_hook("before_evaluate")
            self.run_callback_hooks("before_evaluate", self)

            # evaluate run
            self._evaluate(dataset)

            self.run_default_hook("after_evaluate")
            self.run_callback_hooks("after_evaluate", self)

        except (KeyboardInterrupt, SystemExit) as e:
            self.run_default_hook("on_exception_stopped", e)
            self.run_callback_hooks("on_exception_stopped", self, e)
            raise e
        except Exception as e:
            self.run_default_hook("on_exception_failed", e)
            self.run_callback_hooks("on_exception_failed", self, e)
            if self.evaluate_file.exists():
                io.remove_file(self.evaluate_file)
            raise e
        finally:
            self.run_default_hook("teardown")
            self.run_callback_hooks("teardown", self)

        return self.result

    def _load_dataset(self, dataset: Union[str, Path], dataset_root_dir: str) -> Dataset:
        if Path(dataset).exists():
            dataset = Path(dataset)
            dataset = Dataset.load(name=dataset.parts[-1], root_dir=dataset.parents[0].absolute())
        elif dataset in Dataset.get_dataset_list(dataset_root_dir):
            dataset = Dataset.load(name=dataset, root_dir=dataset_root_dir)
        else:
            raise FileNotFoundError(f"Dataset {dataset} is not exist.")
        return dataset

    def _evaluate(self, dataset: Dataset):
        self.run_default_hook("on_evaluate_start")
        self.run_callback_hooks("on_evaluate_start", self)

        device = self.cfg.device
        model = self.model.to(device)
        dataloader = get_dataset_class("dataset")(
            dataset,
            self.cfg.image_size,
            letter_box=self.cfg.letter_box,
            set_name=self.cfg.set_name,
        ).get_dataloader(self.cfg.batch_size, self.cfg.workers)

        result_parser = get_parser(self.model.task)(
            **self.cfg.to_dict(), categories=dataset.get_categories()
        )

        self.run_default_hook("on_evaluate_loop_start", dataloader)
        self.run_callback_hooks("on_evaluate_loop_start", self, dataloader)

        preds = []
        labels = []
        for i, batch in tqdm.tqdm(enumerate(dataloader, start=1), total=len(dataloader)):
            self.run_default_hook("on_evaluate_step_start", i, batch)
            self.run_callback_hooks("on_evaluate_step_start", self, i, batch)
            images, image_infos, annotations = batch

            result_batch = model(images.to(device))
            result_batch = result_parser(result_batch, image_infos)

            preds.extend(result_batch)
            labels.extend(annotations)

            self.run_default_hook("on_evaluate_step_end", i, batch, result_batch)
            self.run_callback_hooks("on_evaluate_step_end", self, i, batch, result_batch)

        self.run_default_hook("on_evaluate_loop_end", preds)
        self.run_callback_hooks("on_evaluate_loop_end", self, preds)

        metrics = evaluate_function(
            preds,
            labels,
            self.model.task,
            len(dataset.get_categories()),
            image_size=self.cfg.image_size,
        )
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

        self.run_default_hook("on_evaluate_end", result_metrics)
        self.run_callback_hooks("on_evaluate_end", self, result_metrics)

    def is_valid_dataset(self, dataset: Dataset):
        """Check dataset is valid for model.

        Args:
            dataset (Dataset): dataset

        Raises:
            ValueError: if model task is not equal to dataset task
            ValueError: if model categories is not equal to dataset categories
        """
        # check task
        if self.model.task != dataset.task:
            raise ValueError(
                f"Model task {self.model.task} is not equal to dataset task {dataset.task}"
            )

        # check categories
        if [category.name for category in self.model.categories] != dataset.get_category_names():
            raise ValueError(
                f"Model categories {self.model.categories} is not equal to dataset categories {dataset.get_category_names()}"
            )
