"""
Set of types for waffle_dough
"""
from .color_type import ColorType
from .data_type import DataType
from .task_type import TaskType


def get_data_types():
    return list(map(lambda x: x.value, list(DataType)))


def get_task_types():
    return list(map(lambda x: x.value, list(TaskType)))


def get_color_types():
    return list(map(lambda x: x.value, list(ColorType)))


__all__ = [
    "DataType",
    "TaskType",
    "ColorType",
    "get_data_types",
    "get_task_types",
    "get_color_types",
]
