from pathlib import Path

from waffle_hub.hub.evaluator.evaluator import Evaluator
from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.type import task_type


def test_cls_evalutator(cls_test_model: ModelWrapper, tmpdir: Path):
    evaluator = Evaluator(
        root_dir=tmpdir,
        model=cls_test_model,
        task=task_type.CLASSIFICATION,
    )
