__version__ = "0.1.0"

_backends = {
    "ultralytics": ["8.0.25", "8.0.26"],
    "tx_model": ["0.1.0", "0.2.0"],
}


def get_backends() -> dict:
    return _backends
