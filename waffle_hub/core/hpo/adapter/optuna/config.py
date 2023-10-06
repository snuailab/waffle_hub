import enum

from waffle_hub import BaseEnum, DataType, OrderedDict


class SamplerType(BaseEnum):
    RANDOMSAMPLER = enum.auto()
    GRIDSAMPLER = enum.auto()
    TPESAMPLER = enum.auto()


class PrunerType(BaseEnum):
    MEDIANPRUNER = enum.auto()
    PERCENTILEPRUNER = enum.auto()
    SUCCESSIVEHALVINGPRUNER = enum.auto()
    NOPRUNER = enum.auto()


class SearchSpaceType(BaseEnum):
    SUGGEST_CATEGORICAL = enum.auto()
    SUGGEST_INT = enum.auto()
    SUGGEST_FLOAT = enum.auto()


SAMPLER_MAP = OrderedDict(
    {
        SamplerType.RANDOMSAMPLER: {
            "import_path": "optuna.samplers",
            "class_name": "RandomSampler",
        },
        SamplerType.GRIDSAMPLER: {
            "import_path": "optuna.samplers",
            "class_name": "GridSampler",
        },
        SamplerType.TPESAMPLER: {
            "import_path": "optuna.samplers",
            "class_name": "TPESampler",
        },
    }
)

PRUNER_MAP = OrderedDict(
    {
        PrunerType.MEDIANPRUNER: {
            "import_path": "optuna.pruners",
            "class_name": "MedianPruner",
        },
        PrunerType.SUCCESSIVEHALVINGPRUNER: {
            "import_path": "optuna.pruners",
            "class_name": "SuccessiveHalvingPruner",
        },
        PrunerType.NOPRUNER: {
            "import_path": "optuna.pruners",
            "class_name": "NopPruner",
        },
    }
)

for key in list(SAMPLER_MAP.keys()):
    SAMPLER_MAP[str(key).lower()] = SAMPLER_MAP[key]
    SAMPLER_MAP[str(key).upper()] = SAMPLER_MAP[key]

for key in list(PRUNER_MAP.keys()):
    PRUNER_MAP[str(key).lower()] = PRUNER_MAP[key]
    PRUNER_MAP[str(key).upper()] = PRUNER_MAP[key]

SEARCH_SPACE_MAP = OrderedDict(
    {
        SearchSpaceType.SUGGEST_INT: {
            "method_name": "suggest_int",
        },
        SearchSpaceType.SUGGEST_FLOAT: {
            "method_name": "suggest_float",
        },
        SearchSpaceType.SUGGEST_CATEGORICAL: {
            "method_name": "suggest_categorical",
        },
    }
)

for key in list(SEARCH_SPACE_MAP.keys()):
    SEARCH_SPACE_MAP[str(key).lower()] = SEARCH_SPACE_MAP[key]
    SEARCH_SPACE_MAP[str(key).upper()] = SEARCH_SPACE_MAP[key]
