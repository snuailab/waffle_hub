import warnings
from pathlib import Path

from autocare_dlt.tools import train
from waffle_utils.callback import BaseCallback
from waffle_utils.file import io

from waffle_hub.hub.manager.adapter.autocare_dlt.configs import (
    get_data_config,
    get_model_config,
)
from waffle_hub.hub.manager.base_manager import BaseManager
from waffle_hub.type import TaskType


class AutocareDLTCallback(BaseCallback):
    def __init__(self):
        pass

    def on_train_start(self, manager: BaseManager) -> None:
        """Called when the train function begins."""
        # set data
        manager.cfg.dataset_path: Path = Path(manager.cfg.dataset_path)
        train_coco_file = manager.cfg.dataset_path / "train.json"
        val_coco_file = manager.cfg.dataset_path / "val.json"
        test_coco_file = manager.cfg.dataset_path / "test.json"
        data_config = get_data_config(
            manager.DATA_TYPE_MAP[manager.task],
            manager.cfg.image_size
            if isinstance(manager.cfg.image_size, list)
            else [manager.cfg.image_size, manager.cfg.image_size],
            manager.cfg.batch_size,
            manager.cfg.workers,
            str(train_coco_file),
            str(manager.cfg.dataset_path / "images"),
            str(val_coco_file),
            str(manager.cfg.dataset_path / "images"),
            str(test_coco_file) if test_coco_file.exists() else str(val_coco_file),
            str(manager.cfg.dataset_path / "images"),
        )
        if manager.model_type == "LicencePlateRecognition":
            data_config["data"]["mode"] = "lpr"
        if manager.model_type == "Segmenter":
            data_config["data"]["gray"] = True

        manager.cfg.data_config = manager.artifacts_dir / "data.json"
        io.save_json(data_config, manager.cfg.data_config, create_directory=True)

        if manager.task == TaskType.CLASSIFICATION:
            super_cat = [[c.supercategory, c.name] for c in manager.categories]
            super_cat_dict = {}
            for super_cat, cat in super_cat:
                if super_cat not in super_cat_dict:
                    super_cat_dict[super_cat] = []
                super_cat_dict[super_cat].append(cat)

            categories = []
            for super_cat, cat in super_cat_dict.items():
                categories.append({super_cat: cat})

        else:
            categories = manager.get_category_names()

        model_config = get_model_config(
            manager.model_type,
            manager.model_size,
            categories,
            manager.cfg.seed,
            manager.cfg.learning_rate,
            manager.cfg.letter_box,
            manager.cfg.epochs,
        )

        manager.cfg.model_config = manager.artifacts_dir / "model.json"
        io.save_json(model_config, manager.cfg.model_config, create_directory=True)

        # pretrained model
        manager.cfg.pretrained_model = (
            manager.cfg.pretrained_model
            if manager.cfg.pretrained_model is not None
            else manager.WEIGHT_PATH[manager.task][manager.model_type][manager.model_size]
        )
        if not Path(manager.cfg.pretrained_model).exists():
            manager.cfg.pretrained_model = None
            warnings.warn(f"{manager.cfg.pretrained_model} does not exists. Train from scratch.")

        manager.cfg.dataset_path = str(manager.cfg.dataset_path.absolute())

    def training(self, manager: BaseManager) -> None:
        """Called when the training"""
        results = train.run(
            exp_name="train",
            model_cfg=str(manager.cfg.model_config),
            data_cfg=str(manager.cfg.data_config),
            gpus="-1" if manager.cfg.device == "cpu" else str(manager.cfg.device),
            output_dir=str(manager.artifacts_dir),
            ckpt=manager.cfg.pretrained_model,
            overwrite=True,
        )
        if results is None:
            raise RuntimeError("Training failed")
        del results

    def on_train_end(self, manager: BaseManager) -> None:
        """Called when the train function ends."""
        best_ckpt_path = manager.artifacts_dir / "train" / "best_ckpt.pth"
        last_epoch_ckpt_path = manager.artifacts_dir / "train" / "last_epoch_ckpt.pth"
        model_json_path = manager.artifacts_dir / "model.json"

        if best_ckpt_path.exists():
            io.copy_file(best_ckpt_path, manager.best_ckpt_file, create_directory=True)
        if last_epoch_ckpt_path.exists():
            io.copy_file(last_epoch_ckpt_path, manager.last_ckpt_file, create_directory=True)
        if model_json_path.exists():
            io.copy_file(model_json_path, manager.model_json_output_path, create_directory=True)

        metrics = manager.get_metrics()
        if metrics:
            io.save_json(metrics, manager.metric_file, create_directory=True)
