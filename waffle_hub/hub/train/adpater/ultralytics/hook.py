from pathlib import Path

from waffle_utils.callback import BaseCallback
from waffle_utils.file import io

from waffle_hub import TrainStatus
from waffle_hub.hub.train.hook import BaseTrainHook
from waffle_hub.type import TaskType
from waffle_hub.utils.metric_logger import MetricLogger
from waffle_hub.utils.process import run_python_file


class UltralyticsTrainHook(BaseTrainHook):
    def __init__(self, callbacks: list[BaseCallback] = None):
        super().__init__(callbacks)

    def setup(self) -> None:
        """Called when worker starts."""
        self.metric_logger = MetricLogger(
            name=self.name,
            log_dir=self.train_log_dir,
            func=self.get_metrics,
            interval=10,
            prefix="waffle",
            state=self.state,
            state_save_path=self.state_save_path,
        )

    def teardown(self) -> None:
        """Called when worker ends."""
        self.metric_logger.stop()

    def before_train(self) -> None:
        """Called when the train begins."""
        self.state.total_step = self.cfg.epochs
        self.state.current_step = 0

    def on_train_start(self) -> None:
        """Called when the train function begins."""
        self.state.status = TrainStatus.RUNNING
        self.metric_logger.start()
        # set data
        self.cfg.dataset_path: Path = Path(self.cfg.dataset_path)
        if self.backend_task_name in ["detect", "segment"]:
            if self.cfg.dataset_path.suffix not in [".yml", ".yaml"]:
                yaml_files = list(self.cfg.dataset_path.glob("*.yaml")) + list(
                    self.cfg.dataset_path.glob("*.yml")
                )
                if len(yaml_files) != 1:
                    raise FileNotFoundError(f"Ambiguous data file. Detected files: {yaml_files}")
                self.cfg.dataset_path = Path(yaml_files[0]).absolute()
            else:
                self.cfg.dataset_path = self.cfg.dataset_path.absolute()
        elif self.backend_task_name == "classify":
            if not self.cfg.dataset_path.is_dir():
                raise ValueError(
                    f"Classification dataset should be directory. Not {self.cfg.dataset_path}"
                )
            self.cfg.dataset_path = self.cfg.dataset_path.absolute()
        self.cfg.dataset_path = str(self.cfg.dataset_path)

        # pretrained model
        self.cfg.pretrained_model = (
            self.cfg.pretrained_model
            if self.cfg.pretrained_model
            else self.MODEL_TYPES[self.task][self.model_type][self.model_size]
        )

        # other
        if self.task in [TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]:
            self.cfg.letter_box = True  # TODO: hard coding
            # logger.warning(
            #     "letter_box False is not supported for Object Detection and Segmentation."
            # )

    def training(self):
        params = {
            "data": str(self.cfg.dataset_path).replace("\\", "/"),
            "epochs": self.cfg.epochs,
            "batch": self.cfg.batch_size,
            "imgsz": self.cfg.image_size,
            "lr0": self.cfg.learning_rate,
            "lrf": self.cfg.learning_rate,
            "rect": False,  # TODO: hard coding for mosaic
            "device": self.cfg.device,
            "workers": self.cfg.workers,
            "seed": self.cfg.seed,
            "verbose": self.cfg.verbose,
            "project": str(self.root_dir),
            "name": str(self.ARTIFACTS_DIR),
        }
        params.update(self.cfg.advance_params)

        code = f"""if __name__ == "__main__":
        from ultralytics import YOLO
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        try:
            model = YOLO("{self.cfg.pretrained_model}", task="{self.backend_task_name}")
            model.train(
                **{params}
            )
        except Exception as e:
            print(e)
            raise e
        """

        script_file = str((self.root_dir / "ultralytics_train.py").absolute())
        with open(script_file, "w") as f:
            f.write(code)

        run_python_file(script_file)

    def on_train_end(self) -> None:
        """Called when the train function ends."""
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

    def after_train(self) -> None:
        """Called when the train ends."""
        self.state.status = TrainStatus.SUCCESS

    def on_exception_stopped(self, e: Exception) -> None:
        """Called when SIGTERM or SIGINT occurs"""
        self.state.status = TrainStatus.STOPPED
        self.state.set_error(e)

    def on_exception_failed(self, e: Exception) -> None:
        """Called when an error occurs"""
        self.state.status = TrainStatus.FAILED
        self.state.set_error(e)
