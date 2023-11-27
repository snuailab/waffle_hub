import threading
import time
import warnings
from pathlib import Path
from typing import Union

import cpuinfo
import cv2
import torch
import tqdm
from torch import nn
from waffle_utils.file import io
from waffle_utils.image.io import save_image
from waffle_utils.utils import type_validator
from waffle_utils.video.io import create_video_writer

from waffle_dough.type.task_type import TaskType
from waffle_hub.dataset import Dataset
from waffle_hub.hub.model.result_parser import get_parser
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import InferenceConfig, TrainConfig
from waffle_hub.schema.fields.category import Category
from waffle_hub.schema.result import InferenceResult
from waffle_hub.utils.callback import InferenceCallback
from waffle_hub.utils.data import IMAGE_EXTS, VIDEO_EXTS, get_dataset_class
from waffle_hub.utils.draw import draw_results
from waffle_hub.utils.memory import device_context


class Inferencer:
    """
    Inference manager class
    """

    # directory settting
    INFERENCE_DIR = Path("inferences")
    DRAW_DIR = INFERENCE_DIR / Path("draws")

    # inference results file path ###--
    INFERENCE_FILE = INFERENCE_DIR / "inferences.json"

    def __init__(
        self,
        root_dir: Path,
        model: Union[ModelWrapper, nn.Module],
        task: Union[str, TaskType],
        categories: list[Union[str, int, float, dict, Category]],
        train_config: TrainConfig = None,
    ):
        self.root_dir = Path(root_dir)
        self.model = model
        self.task = task
        self.categories = categories
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
    def inference_dir(self) -> Path:
        """Inference Results Directory"""
        return self.root_dir / self.INFERENCE_DIR

    @property
    def inference_file(self) -> Path:
        """Inference Results File path"""
        return self.root_dir / self.INFERENCE_FILE

    @property
    def draw_dir(self) -> Path:
        """Draw Results Directory"""
        return self.root_dir / self.DRAW_DIR

    @classmethod
    def get_inference_result(cls, root_dir: Union[str, Path]) -> list[dict]:
        """Get inference result from inference file.

        Args:
            root_dir (Union[str, Path]): root directory of inference file

        Example:
            >>> Inferencer.get_inference_result(root_dir)
            [
                {
                    "id": "00000001",
                    "category": "person",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "score": 0.9,
                },
            ]

        Returns:
            list[dict]: inference result
        """
        inference_file_path = Path(root_dir) / cls.INFERENCE_FILE
        if not inference_file_path.exists():
            warnings.warn(f"inference file {inference_file_path} is not exist. Inference First.")
            return []
        return io.load_json(inference_file_path)

    # methods
    def inference(
        self,
        source: Union[str, Dataset],
        recursive: bool = True,
        image_size: Union[int, list[int]] = None,
        letter_box: bool = None,
        batch_size: int = 4,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        half: bool = False,
        workers: int = 2,
        device: str = "0",
        draw: bool = False,
        show: bool = False,
        hold: bool = True,
    ) -> InferenceResult:
        """Start Inference

        Args:
            source (str): image directory or image path or video path.
            recursive (bool, optional): recursive. Defaults to True.
            image_size (Union[int, list[int]], optional): image size. If None, use train config or defaults to 224.
            letter_box (bool, optional): letter box. If None, use train config or defaults to True.
            batch_size (int, optional): batch size. Defaults to 4.
            confidence_threshold (float, optional): confidence threshold. Not required in classification. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Not required in classification. Defaults to 0.5.
            half (bool, optional): half. Defaults to False.
            workers (int, optional): workers. Defaults to 2.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
            draw (bool, optional): draw. Defaults to False.
            show (bool, optional): show. Defaults to False.
            hold (bool, optional): hold. Defaults to True.


        Raises:
            FileNotFoundError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

        Example:
            >>> inferencer = Inferencer(...)
            >>> inference_result = hub.inference(
                    source="path/to/images",
                    batch_size=4,
                    image_size=640,
                    letterbox=False,
                    confidence_threshold=0.25,
                    iou_threshold=0.5,
                    workers=4,
                    device="0",
                    draw=True,
                )
            # or simply use train option by passing None
            >>> inferencer = Inferencer(... , train_config=train_config)
            >>> inference_result = hub.inference(
                    ...
                    image_size=None,  # use train option or default to 224
                    letterbox=None,  # use train option or default to True
                    ...
                )
            >>> inference_result.predictions
            [{"relative/path/to/image/file": [{"category": "1", "bbox": [0, 0, 100, 100], "score": 0.9}, ...]}, ...]

        Returns:
            InferenceResult: inference result
        """

        @device_context("cpu" if device == "cpu" else device)
        def inner(callback: InferenceCallback, result: InferenceResult):
            try:
                self.before_inference()
                self.on_inference_start()
                self.inferencing(callback)
                self.on_inference_end()
                self.after_inference(result)
                callback.force_finish()
            except Exception as e:
                if self.inference_dir.exists():
                    io.remove_directory(self.inference_dir)
                callback.force_finish()
                callback.set_failed()
                raise e

        # image_dir, image_path, video_path
        if isinstance(source, (str, Path)):
            if Path(source).exists():
                source = Path(source)
                if source.is_dir() or source.suffix in IMAGE_EXTS:
                    source = source.absolute()
                    source_type = "image"
                elif source.suffix in VIDEO_EXTS:
                    source = str(source.absolute())
                    source_type = "video"
                else:
                    raise ValueError(
                        f"Invalid source: {source}\n"
                        + "Please use image directory or image path or video path."
                    )
            else:
                raise FileNotFoundError(f"Source {source} is not exist.")
        else:
            raise ValueError(
                f"Invalid source: {source}\n"
                + "Please use image directory or image path or video path."
            )

        # overwrite training config or default
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

        self.inference_cfg = InferenceConfig(
            source=source,
            source_type=source_type,
            batch_size=batch_size,
            recursive=recursive,
            image_size=image_size if isinstance(image_size, list) else [image_size, image_size],
            letter_box=letter_box,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            half=half,
            workers=workers,
            device="cpu" if device == "cpu" else f"cuda:{device}",
            draw=draw or show,
            show=show,
        )

        callback = InferenceCallback(100)  # dummy step
        result = InferenceResult()
        result.callback = callback
        if hold:
            inner(callback, result)
        else:
            thread = threading.Thread(target=inner, args=(callback, result), daemon=True)
            callback.register_thread(thread)
            callback.start()

        return result

    def benchmark(
        self,
        image_size: Union[int, list[int]] = None,
        batch_size: int = 16,
        device: str = "0",
        half: bool = False,
        trial: int = 100,
    ) -> dict:
        """Benchmark Model

        Args:
            image_size (Union[int, list[int]], optional): Inference image size. If None, same train config (recommended) or defaults to 224.
            batch_size (int, optional): dynamic batch size. Defaults to 16.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
            half (bool, optional): half. Defaults to False.
            trial (int, optional): number of trials. Defaults to 100.

        Example:
            >>> inferencer = Inferencer(...)
            >>> inferencer.benchmark(
                    image_size=640,
                    batch_size=16,
                    device="0",
                    half=False,
                    trial=100,
                )
            {
                "inference_time": 0.123,
                "fps": 123.123,
                "image_size": [640, 640],
                "batch_size": 16,
                "device": "0",
                "cpu_name": "Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz",
                "gpu_name": "GeForce GTX 1080 Ti",
            }

        Returns:
            dict: benchmark result
        """

        if half and (not torch.cuda.is_available() or device == "cpu"):
            raise RuntimeError("half is not supported in cpu")

        # overwrite training config or default
        if image_size is None:
            if self.train_config is not None:
                image_size = self.train_config.image_size
            else:
                image_size = 224  # default image size
        image_size = [image_size, image_size] if isinstance(image_size, int) else image_size

        device = "cpu" if device == "cpu" else f"cuda:{device}"

        self.model = self.model.to(device) if not half else self.model.half().to(device)

        dummy_input = torch.randn(
            batch_size, 3, *image_size, dtype=torch.float32 if not half else torch.float16
        )
        dummy_input = dummy_input.to(device)

        self.model.eval()
        with torch.no_grad():
            start = time.time()
            for _ in tqdm.tqdm(range(trial)):
                self.model(dummy_input)
            end = time.time()
            inference_time = end - start

        return {
            "inference_time": inference_time,
            # image throughput per second
            "fps": trial * batch_size / inference_time,
            "image_size": image_size,
            "batch_size": batch_size,
            "precision": "fp16" if half else "fp32",
            "device": device,
            "cpu_name": cpuinfo.get_cpu_info()["brand_raw"],
            "gpu_name": torch.cuda.get_device_name(0) if device != "cpu" else None,
        }

    # inference hooks
    def before_inference(self):
        pass

    def on_inference_start(self):
        pass

    def inferencing(self, callback: InferenceCallback) -> str:
        device = self.inference_cfg.device
        model = self.model.to(device)
        result_parser = get_parser(self.task)(
            **self.inference_cfg.to_dict(), categories=self.categories
        )

        if self.inference_cfg.source_type == "image":
            dataset = get_dataset_class(self.inference_cfg.source_type)(
                self.inference_cfg.source,
                self.inference_cfg.image_size,
                letter_box=self.inference_cfg.letter_box,
                recursive=self.inference_cfg.recursive,
            )
            dataloader = dataset.get_dataloader(
                self.inference_cfg.batch_size, self.inference_cfg.workers
            )
        elif self.inference_cfg.source_type == "video":
            dataset = get_dataset_class(self.inference_cfg.source_type)(
                self.inference_cfg.source,
                self.inference_cfg.image_size,
                letter_box=self.inference_cfg.letter_box,
            )
            dataloader = dataset.get_dataloader(
                self.inference_cfg.batch_size, self.inference_cfg.workers
            )
        else:
            raise ValueError(f"Invalid source type: {self.inference_cfg.source_type}")

        if self.inference_cfg.draw and self.inference_cfg.source_type == "video":
            writer = None

        results = []
        callback._total_steps = len(dataloader) + 1
        for i, (images, image_infos) in tqdm.tqdm(
            enumerate(dataloader, start=1), total=len(dataloader)
        ):
            result_batch = model(images.to(device))
            result_batch = result_parser(result_batch, image_infos)
            for result, image_info in zip(result_batch, image_infos):

                results.append({str(image_info.image_rel_path): [res.to_dict() for res in result]})

                if self.inference_cfg.draw:
                    io.make_directory(self.draw_dir)
                    draw = draw_results(
                        image_info.ori_image,
                        result,
                        names=[x["name"] for x in self.categories],
                    )

                    if self.inference_cfg.source_type == "video":
                        if writer is None:
                            h, w = draw.shape[:2]
                            writer = create_video_writer(
                                str(
                                    self.inference_dir
                                    / Path(self.inference_cfg.source).with_suffix(".mp4").name
                                ),
                                dataset.fps,
                                (w, h),
                            )
                        writer.write(draw)

                        draw_path = (
                            self.draw_dir
                            / Path(self.inference_cfg.source).stem
                            / Path(image_info.image_rel_path).with_suffix(".png")
                        )
                    else:
                        draw_path = self.draw_dir / Path(image_info.image_rel_path).with_suffix(
                            ".png"
                        )
                    save_image(draw_path, draw, create_directory=True)

                if self.inference_cfg.show:
                    if not self.inference_cfg.draw:
                        draw = draw_results(
                            image_info.ori_image,
                            result,
                            names=[x["name"] for x in self.categories],
                        )
                    cv2.imshow("result", draw)
                    cv2.waitKey(1)

            callback.update(i)

        if self.inference_cfg.draw and self.inference_cfg.source_type == "video":
            writer.release()

        if self.inference_cfg.show:
            cv2.destroyAllWindows()

        io.save_json(
            results,
            self.inference_file,
            create_directory=True,
        )

    def on_inference_end(self):
        pass

    def after_inference(self, result: InferenceResult):
        result.predictions = self.get_inference_result(self.root_dir)
        if self.inference_cfg.draw:
            result.draw_dir = self.draw_dir
