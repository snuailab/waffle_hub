from enum import Enum


class HPOMethod(Enum):
    RANDOMSAMPLER = "RANDOMSAMPLER"
    GRIDSAMPLER = "GRIDSAMPLER"
    BOHBSAMPLER = "BOHBSAMPLER"
    TPESAMPLER = "TPESAMPLER"


class SearchOption(Enum):
    FAST = "FAST"
    MEDIUM = "MEDIUM"
    LONG = "LONG"


class Objective(Enum):
    ACCURACY = "ACCURACY"
    LOSS = "LOSS"
