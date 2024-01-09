from pathlib import Path

from waffle_utils.file import io

from waffle_hub.hub.manager.base_manager import BaseManager
from waffle_hub.hub.manager.callbacks import BaseTrainCallback
from waffle_hub.type import TaskType
from waffle_hub.utils.process import run_python_file


class UltralyticsTrainCallback(BaseTrainCallback):
    def __init__(self):
        pass

    def on_train_start(self, manager: BaseManager) -> None:
        """Called when the train function begins."""
        # set data
        manager.cfg.dataset_path: Path = Path(manager.cfg.dataset_path)
        if manager.backend_task_name in ["detect", "segment"]:
            if manager.cfg.dataset_path.suffix not in [".yml", ".yaml"]:
                yaml_files = list(manager.cfg.dataset_path.glob("*.yaml")) + list(
                    manager.cfg.dataset_path.glob("*.yml")
                )
                if len(yaml_files) != 1:
                    raise FileNotFoundError(f"Ambiguous data file. Detected files: {yaml_files}")
                manager.cfg.dataset_path = Path(yaml_files[0]).absolute()
            else:
                manager.cfg.dataset_path = manager.cfg.dataset_path.absolute()
        elif manager.backend_task_name == "classify":
            if not manager.cfg.dataset_path.is_dir():
                raise ValueError(
                    f"Classification dataset should be directory. Not {manager.cfg.dataset_path}"
                )
            manager.cfg.dataset_path = manager.cfg.dataset_path.absolute()
        manager.cfg.dataset_path = str(manager.cfg.dataset_path)

        # pretrained model
        manager.cfg.pretrained_model = (
            manager.cfg.pretrained_model
            if manager.cfg.pretrained_model
            else manager.MODEL_TYPES[manager.task][manager.model_type][manager.model_size]
        )

        # other
        if manager.task in [TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]:
            manager.cfg.letter_box = True  # TODO: hard coding
            # logger.warning(
            #     "letter_box False is not supported for Object Detection and Segmentation."
            # )

        manager.save_train_config(manager.cfg, manager.train_config_file)

    def training(self, manager: BaseManager) -> None:
        params = {
            "data": str(manager.cfg.dataset_path).replace("\\", "/"),
            "epochs": manager.cfg.epochs,
            "batch": manager.cfg.batch_size,
            "imgsz": manager.cfg.image_size,
            "lr0": manager.cfg.learning_rate,
            "lrf": manager.cfg.learning_rate,
            "rect": False,  # TODO: hard coding for mosaic
            "device": manager.cfg.device,
            "workers": manager.cfg.workers,
            "seed": manager.cfg.seed,
            "verbose": manager.cfg.verbose,
            "project": str(manager.root_dir),
            "name": str(manager.ARTIFACTS_DIR),
        }
        params.update(manager.cfg.advance_params)

        code = f"""if __name__ == "__main__":
        from ultralytics import YOLO
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        try:
            model = YOLO("{manager.cfg.pretrained_model}", task="{manager.backend_task_name}")
            model.train(
                **{params}
            )
        except Exception as e:
            print(e)
            raise e
        """

        script_file = str((manager.root_dir / "ultralytics_train.py").absolute())
        with open(script_file, "w") as f:
            f.write(code)

        run_python_file(script_file)

    def on_train_end(self, manager: BaseManager) -> None:
        """Called when the train function ends."""
        io.copy_file(
            manager.artifacts_dir / "weights" / "best.pt",
            manager.best_ckpt_file,
            create_directory=True,
        )
        io.copy_file(
            manager.artifacts_dir / "weights" / "last.pt",
            manager.last_ckpt_file,
            create_directory=True,
        )
        io.save_json(manager.get_metrics(), manager.metric_file)
