__version__ = "0.1.0"


from collections import OrderedDict

_backends = OrderedDict(
    {
        "ultralytics": ["8.0.25"],
    }
)


def get_backends() -> dict:
    return _backends
