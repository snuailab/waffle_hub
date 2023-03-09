from .base_hub import Hub
from .ultralytics_hub import UltralyticsHub

__all__ = ["Hub", "UltralyticsHub"]

HUB_MAP = {"ultralytics": UltralyticsHub}
