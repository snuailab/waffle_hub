"""
Base Type Class

>> TaskType.CLASSIFICATION
classification
>> TaskType.CLASSIFICATION.upper()
CLASSIFICATION
>> TaskType.CLASSIFICATION.name
'CLASSIFICATION'
>> TaskType.CLASSIFICATION == "classification"
True
>> TaskType.CLASSIFICATION == "CLASSIFICATION"
True
>> "classification" in list(TaskType)
True
"""

from enum import Enum, EnumMeta
from typing import Any

from typing_extensions import SupportsIndex


class CustomEnumMeta(str):
    def __eq__(self, other):
        if hasattr(other, "value"):
            other = other.value
        return self.lower() == str(other).lower()

    def __ne__(self, other):
        if hasattr(other, "value"):
            other = other.value
        return self.lower() != str(other).lower()

    def __hash__(self):
        return hash(self.lower())

    def __repr__(self):
        return self.lower()

    def __str__(self):
        return self.lower()

    def __reduce__(self):
        return self.lower()  # for pickle

    def __reduce_ex__(self, protocol):
        return self.lower()

    def __call__(self):
        return self.lower()

    def __contains__(self, item):
        return item in self.value


class CaseInsensitiveEnumMeta(EnumMeta):
    def __getitem__(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member
        raise KeyError(f"No such member: {name}")


class BaseType(CustomEnumMeta, Enum, metaclass=CaseInsensitiveEnumMeta):
    def __hash__(self):
        return hash(self.value)
