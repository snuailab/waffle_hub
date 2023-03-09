from .base_hub import BaseHub
from .ultralytics_hub import UltralyticsHub

__all__ = ["BaseHub", "UltralyticsHub"]

HUB_MAP = {"ultralytics": UltralyticsHub}
