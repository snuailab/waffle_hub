import time
import warnings
from pathlib import Path
from typing import Union

import cpuinfo
import torch
import tqdm
from waffle_utils.callback import BaseCallback
from waffle_utils.file import io

from waffle_hub import InferenceStatus
from waffle_hub.dataset import Dataset
from waffle_hub.hub.inferencer.hook import BaseInferenceHook
from waffle_hub.hub.model.result_parser import get_parser
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.schema.configs import InferenceConfig
from waffle_hub.schema.result import InferenceResult
from waffle_hub.schema.state import InferenceState
from waffle_hub.utils.data import IMAGE_EXTS, VIDEO_EXTS, get_dataset_class
from waffle_hub.utils.memory import device_context


class Inferencer(BaseInferenceHook):
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
        model: ModelWrapper,
        callbacks: list[BaseCallback] = None,
    ):
        super().__init__(callbacks)
        self.root_dir = Path(root_dir)
        self.model = model
        self.state = InferenceState(status=InferenceStatus.INIT)
        self.result = InferenceResult()

    # properties
    @property
    def inference_dir(self) -> Path:
        """Inference Results Directory"""
        return self.root_dir / self.INFERENCE_DIR

    @property
    def draw_dir(self) -> Path:
        """Draw Results Directory"""
        return self.root_dir / self.DRAW_DIR

    @property
    def inference_file(self) -> Path:
        """Inference Results File path"""
        return self.root_dir / self.INFERENCE_FILE

    @classmethod
    def get_inference_result(cls, root_dir: Union[str, Path]) -> list[dict]:
        """Get inference result from inference file.

        Args:
            root_dir (Union[str, Path]): root directory of inference file

        Examples:
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
    @device_context
    def inference(
        self,
        source: Union[str, Path],
        recursive: bool = True,
        image_size: Union[int, list[int]] = 224,
        letter_box: bool = True,
        batch_size: int = 4,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        half: bool = False,
        workers: int = 2,
        device: str = "0",
        draw: bool = False,
        show: bool = False,
    ) -> InferenceResult:
        """Start Inference

        Args:
            source (Union[str, Path]): image directory or image path or video path.
            recursive (bool, optional): recursive. Defaults to True.
            image_size (Union[int, list[int]], optional): image size. Defaults to 224.
            letter_box (bool, optional): letter box. Defaults to True.
            batch_size (int, optional): batch size. Defaults to 4.
            confidence_threshold (float, optional): confidence threshold. Not required in classification. Defaults to 0.25.
            iou_threshold (float, optional): iou threshold. Not required in classification. Defaults to 0.5.
            half (bool, optional): half. Defaults to False.
            workers (int, optional): workers. Defaults to 2.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
            draw (bool, optional): register draw callback. Defaults to False.
            show (bool, optional): register show callback. Defaults to False.

        Raises:
            FileNotFoundError: if can not detect appropriate dataset.
            e: something gone wrong with ultralytics

        Examples:
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
                )
            >>> inference_result.predictions
            [{"relative/path/to/image/file": [{"category": "1", "bbox": [0, 0, 100, 100], "score": 0.9}, ...]}, ...]

        Returns:
            InferenceResult: inference result
        """
        # draw option
        if draw:
            from .callbacks import InferenceDrawCallback

            self.register_callback(InferenceDrawCallback(self.draw_dir))
        # show option
        if show:
            from .callbacks import InferenceShowCallback

            self.register_callback(InferenceShowCallback())

        try:
            self.run_default_hook("setup")
            self.run_callback_hooks("setup", self)

            # inference settings
            # image_dir, image_path, video_path, dataset_name, dataset
            if isinstance(source, (str, Path)):
                if Path(source).exists():
                    source = Path(source)
                    if source.is_dir():
                        source = source.absolute()
                        source_type = "image"
                    elif source.suffix in IMAGE_EXTS:
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

            self.cfg = InferenceConfig(
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
            )
            self.run_default_hook("before_inference")
            self.run_callback_hooks("before_inference", self)

            # run inference
            self._inference()

            self.run_default_hook("after_inference")
            self.run_callback_hooks("after_inference", self)

        except (KeyboardInterrupt, SystemExit) as e:
            self.run_default_hook("on_exception_stopped", e)
            self.run_callback_hooks("on_exception_stopped", self, e)
            raise e
        except Exception as e:
            self.run_default_hook("on_exception_failed", e)
            self.run_callback_hooks("on_exception_failed", self, e)
            # if self.inference_dir.exists():
            #     io.remove_directory(self.inference_dir, recursive=True)
            raise e
        finally:
            self.run_default_hook("teardown")
            self.run_callback_hooks("teardown", self)

        return self.result

    def benchmark(
        self,
        image_size: Union[int, list[int]] = 224,
        batch_size: int = 16,
        device: str = "0",
        half: bool = False,
        trial: int = 100,
    ) -> dict:
        """Benchmark Model

        Args:
            image_size (Union[int, list[int]], optional): Inference image size. Defaults to 224.
            batch_size (int, optional): dynamic batch size. Defaults to 16.
            device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
            half (bool, optional): half. Defaults to False.
            trial (int, optional): number of trials. Defaults to 100.

        Examples:
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

    def _inference(self) -> str:
        self.run_default_hook("on_inference_start")
        self.run_callback_hooks("on_inference_start", self)

        device = self.cfg.device
        model = self.model.to(device)
        result_parser = get_parser(self.model.task)(
            **self.cfg.to_dict(), categories=self.model.categories
        )

        if self.cfg.source_type == "image":
            dataset = get_dataset_class(self.cfg.source_type)(
                self.cfg.source,
                self.cfg.image_size,
                letter_box=self.cfg.letter_box,
                recursive=self.cfg.recursive,
            )
            dataloader = dataset.get_dataloader(self.cfg.batch_size, self.cfg.workers)
        elif self.cfg.source_type == "video":
            dataset = get_dataset_class(self.cfg.source_type)(
                self.cfg.source,
                self.cfg.image_size,
                letter_box=self.cfg.letter_box,
            )
            dataloader = dataset.get_dataloader(self.cfg.batch_size, self.cfg.workers)
        else:
            raise ValueError(f"Invalid source type: {self.cfg.source_type}")

        self.run_default_hook("on_inference_loop_start", dataset, dataloader)
        self.run_callback_hooks("on_inference_loop_start", self, dataset, dataloader)

        results = []
        for i, batch in tqdm.tqdm(enumerate(dataloader, start=1), total=len(dataloader)):
            self.run_default_hook("on_inference_step_start", i, batch)
            self.run_callback_hooks("on_inference_step_start", self, i, batch)
            images, image_infos = batch

            result_batch = model(images.to(device))
            result_batch = result_parser(result_batch, image_infos)

            for result, image_info in zip(result_batch, image_infos):
                results.append({str(image_info.image_rel_path): [res.to_dict() for res in result]})

            self.run_default_hook("on_inference_step_end", i, batch, result_batch)
            self.run_callback_hooks("on_inference_step_end", self, i, batch, result_batch)

        self.run_default_hook("on_inference_loop_end", results)
        self.run_callback_hooks("on_inference_loop_end", self, results)

        self.result.predictions = results
        io.save_json(
            results,
            self.inference_file,
            create_directory=True,
        )

        self.run_default_hook("on_inference_end")
        self.run_callback_hooks("on_inference_end", self)
