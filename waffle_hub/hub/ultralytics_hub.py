"""
Ultralytics Hub
"""

from waffle_hub import get_installed_backend_version

BACKEND_NAME = "ultralytics"
BACKEND_VERSION = get_installed_backend_version(BACKEND_NAME)

from dataclasses import asdict
from pathlib import Path

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
from waffle_utils.file import io

from waffle_hub.schemas.configs import (
    Classes,
    ClassificationPrediction,
    DetectionPrediction,
    Prediction,
    Train,
)

from . import BaseHub


class UltralyticsHub(BaseHub):

    # Common
    TASKS = ["detect", "classify"]  # TODO: segment
    MODEL_TYPES = ["yolov8"]
    MODEL_SIZES = list("nsmlx")  # TODO: generalize

    # Backend Specifics
    TASK_SUFFIX = {
        "detect": "",
        "classify": "-cls",
        "segment": "-seg",
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
        if self.task in ["detect", "segment"]:
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
        elif self.task == "classify":
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
            + self.TASK_SUFFIX[self.task]
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
            model = YOLO(pretrained_model, task=self.task)
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
        recursive: bool = True,
        image_size: int = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        half: bool = False,
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

        # TODO: get images function needed (in waffle_utils)
        def _get_images(d, recursive: bool = True) -> list[str]:
            exp = "**/*" if recursive else "*"
            return list(
                map(
                    str,
                    list(Path(d).glob(exp + ".png"))
                    + list(Path(d).glob(exp + ".jpg"))
                    + list(Path(d).glob(exp + ".PNG"))
                    + list(Path(d).glob(exp + ".JPG")),
                )
            )

        source = Path(source)
        if source.is_file():
            image_paths = [str(source)]
            common_path = ""
        elif source.is_dir():
            image_paths = _get_images(source, recursive=recursive)
            common_path = str(source)
        else:
            raise ValueError(f"Cannot recognize source {source}")

        # overwrite training config
        train_config = io.load_yaml(self.train_config_file)
        image_size = (
            image_size if image_size else train_config.get("image_size")
        )

        try:
            model = YOLO(self.best_ckpt_file, task=self.task)

            results: list[Results] = model.predict(
                source=image_paths,
                imgsz=image_size,
                conf=conf_thres,
                iou=iou_thres,
                half=half,
                device=device,
            )

        except Exception as e:
            raise e

        # parse predictions
        for image_path, result in zip(image_paths, results):
            relpath = str(Path(image_path).relative_to(common_path))

            prediction = asdict(
                Prediction(
                    image_path=relpath,
                    predictions=self.parse_result(
                        result
                    ),  # TODO: cannot move to cpu now. https://github.com/ultralytics/ultralytics/issues/1318
                )
            )

            io.save_json(
                prediction,
                self.inference_dir / Path(relpath).with_suffix(".json"),
                create_directory=True,
            )

        return str(self.inference_dir)

    def evaluation(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    def parse_result(self, result: Results) -> dict:
        """Parse Ultralytics predict results

        Args:
            result (Results): ultralytics prediction output. TODO: check segmentation compatibility.

        Returns:
            dict: result dictionary
        """
        results = []
        if result.boxes is not None:
            for xywh, cls_idx, conf, segment in zip(
                result.boxes.xywh,
                result.boxes.cls,
                result.boxes.conf,
                result.masks.segments
                if result.masks
                else [None] * len(result),
            ):
                results.append(
                    DetectionPrediction(
                        bbox=list(xywh.cpu().numpy().astype(float)),
                        class_name=result.names[int(cls_idx)],
                        confidence=float(conf.cpu()),
                        segment=list(segment.numpy().astype(float))
                        if segment
                        else [],
                    )
                )
        if result.probs is not None:
            for cls_idx, prob in enumerate(result.probs):
                results.append(
                    ClassificationPrediction(
                        score=float(prob.cpu()),
                        class_name=result.names[int(cls_idx)],
                    )
                )
        return results
