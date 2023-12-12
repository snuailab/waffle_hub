import warnings
from abc import ABC
from collections import OrderedDict
from typing import Any, Callable


class BaseHooks(ABC):
    def __init__(self):
        pass


# hooks
def run_hooks(workers: Callable, event: str, *args: Any, **kwargs: Any):
    # workers: Trainer, Evaluator, Inferencer, Exporter
    if not hasattr(workers, "hook_classes"):
        raise AttributeError("The workers must have hook_classes attribute. Initalize_hooks first.")

    for cls_id, hook_cls in workers.hook_classes.items():
        method = getattr(hook_cls, event, None)
        if method is None:
            continue
        if not callable(method):
            warnings.warn(
                f"Skipping the hook {hook_cls.__class__.__name__}, becuase it is not callable."
            )
            continue
        method(*args, **kwargs)


def initalize_hooks(workers: Callable, hook_cls: BaseHooks):
    # workers: Trainer, Evaluator, Inferencer, Exporter

    workers.hook_classes = OrderedDict({0: hook_cls})
    workers.hooks_idx = 1


def register_hook(workers: Callable, hook_cls: BaseHooks):
    # workers: Trainer, Evaluator, Inferencer, Exporter
    if not hasattr(workers, "hook_classes"):
        raise AttributeError("The workers must have hook_classes attribute. Initalize_hooks first.")

    if not isinstance(hook_cls, BaseHooks):
        raise TypeError(f"hook_cls must be subclass of BaseHooks, not {type(hook_cls)}")
    workers.hook_classes[workers.hooks_idx] = hook_cls
    workers.hooks_idx += 1


def delete_hook(workers: Callable, cls_id: int):
    # workers: Trainer, Evaluator, Inferencer, Exporter
    if not hasattr(workers, "hook_classes"):
        raise AttributeError("The workers must have hook_classes attribute. Initalize_hooks first.")

    if cls_id == 0:
        raise ValueError("Default hooks cannot be deleted.")
    workers.hook_classes.pop(cls_id)


def get_hooks(workers: Callable) -> list[tuple[int, str]]:
    # workers: Trainer, Evaluator, Inferencer, Exporter
    if not hasattr(workers, "hook_classes"):
        raise AttributeError("The workers must have hook_classes attribute. Initalize_hooks first.")

    return [
        (cls_id, hook_cls.__class__.__name__) for cls_id, hook_cls in workers.hook_classes.items()
    ]
