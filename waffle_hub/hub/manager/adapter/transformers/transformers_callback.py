import os

from torchvision import transforms as T
from transformers import (
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)
from waffle_utils.file import io

from datasets import load_from_disk
from waffle_hub.hub.manager.adapter.transformers.train_input_helper import (
    ClassifierInputHelper,
    ObjectDetectionInputHelper,
)
from waffle_hub.hub.manager.adapter.transformers.trainer_callback import CustomCallback
from waffle_hub.hub.manager.base_manager import BaseManager
from waffle_hub.hub.manager.callbacks import BaseTrainCallback
from waffle_hub.type import TaskType


class TransformersTrainCallback(BaseTrainCallback):
    def __init__(self):
        pass

    def on_train_start(self, manager: BaseManager) -> None:
        # overwrite train config with default config
        manager.cfg.pretrained_model = manager.MODEL_TYPES[manager.task][manager.model_type][
            manager.model_size
        ]

        dataset = load_from_disk(manager.cfg.dataset_path)

        if manager.task == TaskType.CLASSIFICATION:
            helper = ClassifierInputHelper(manager.cfg.pretrained_model, manager.cfg.image_size)
            manager.cfg.train_input = helper.get_train_input()
            categories = dataset["train"].features["label"].names
            id2label = {index: x for index, x in enumerate(categories, start=0)}
            manager.cfg.train_input.model = AutoModelForImageClassification.from_pretrained(
                manager.cfg.pretrained_model,
                num_labels=len(id2label),
                ignore_mismatched_sizes=True,
            )

        elif manager.task == TaskType.OBJECT_DETECTION:
            helper = ObjectDetectionInputHelper(manager.cfg.pretrained_model, manager.cfg.image_size)
            manager.cfg.train_input = helper.get_train_input()
            categories = dataset["train"].features["objects"].feature["category"].names
            id2label = {index: x for index, x in enumerate(categories, start=0)}
            label2id = {x: index for index, x in enumerate(categories, start=0)}
            manager.cfg.train_input.model = AutoModelForObjectDetection.from_pretrained(
                manager.cfg.pretrained_model,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            raise NotImplementedError

        if manager.cfg.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            manager.cfg.train_input.model = manager.cfg.train_input.model.to("cpu")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = manager.cfg.device

        transforms = helper.get_transforms()
        dataset["train"] = dataset["train"].with_transform(transforms)
        dataset["val"] = dataset["val"].with_transform(transforms)
        manager.cfg.train_input.dataset = dataset

        manager.cfg.train_input.training_args = TrainingArguments(
            output_dir=str(manager.artifacts_dir),
            per_device_train_batch_size=manager.cfg.batch_size,
            num_train_epochs=manager.cfg.epochs,
            remove_unused_columns=False,
            push_to_hub=False,
            logging_strategy="epoch" if manager.cfg.verbose else "no",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=manager.cfg.learning_rate,
            dataloader_num_workers=manager.cfg.workers,
            seed=manager.cfg.seed,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            load_best_model_at_end=False,
            use_cpu=(manager.cfg.device == "cpu"),
        )

    def training(self, manager: BaseManager) -> None:
        trainer = Trainer(
            model=manager.cfg.train_input.model,
            args=manager.cfg.train_input.training_args,
            data_collator=manager.cfg.train_input.collator,
            train_dataset=manager.cfg.train_input.dataset["train"],
            eval_dataset=manager.cfg.train_input.dataset["val"],
            tokenizer=manager.cfg.train_input.image_processor,
            compute_metrics=manager.cfg.train_input.compute_metrics,
        )
        trainer.add_callback(CustomCallback(trainer, manager.metric_file))
        trainer.train()
        trainer.save_model(str(manager.artifacts_dir / "weights" / manager.LAST_CKPT_FILE))
        trainer._load_best_model()
        trainer.save_model(str(manager.artifacts_dir / "weights" / manager.BEST_CKPT_FILE))

    def on_train_end(self, manager: BaseManager) -> None:
        io.copy_files_to_directory(
            manager.artifacts_dir / "weights" / manager.BEST_CKPT_FILE,
            manager.best_ckpt_file,
            create_directory=True,
        )
        io.copy_files_to_directory(
            manager.artifacts_dir / "weights" / manager.LAST_CKPT_FILE,
            manager.last_ckpt_file,
            create_directory=True,
        )
        io.save_json(manager.get_metrics(), manager.metric_file)
