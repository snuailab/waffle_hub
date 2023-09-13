import enum


class HPOEnum(enum.Enum):
    # framework
    OPTUNA = "OPTUNA"
    RAYTUNE = "RAYTUNE"
    # search option
    FAST = "FAST"
    MEDIUM = "MEDIUM"
    LONG = "LONG"
    # sampler
    RANDOMSAMPLER = "RANDOMSAMPLER"
    GRIDSAMPLER = "GRIDSAMPLER"
    BOHBSAMPLER = "BOHBSAMPLER"
    TPESAMPLER = "TPESAMPLER"
    # objective
    ACCURACY = "ACCURACY"
    LOSS = "LOSS"

    def get_framework():
        return (HPOEnum.OPTUNA, HPOEnum.RAYTUNE)

    def get_search_option():
        return (HPOEnum.FAST, HPOEnum.MEDIUM, HPOEnum.LONG)

    def get_sampler():
        return (HPOEnum.RANDOMSAMPLER, HPOEnum.GRIDSAMPLER, HPOEnum.BOHBSAMPLER, HPOEnum.TPESAMPLER)

    def get_objective():
        return (HPOEnum.ACCURACY, HPOEnum.LOSS)


__all__ = ["HPOEnum", "OptunaHPO", "RaytuneHPO"]
