import warnings

from waffle_hub import get_backends

try:
    import ultralytics

    versions = get_backends()["ultralytics"]
    if ultralytics.__version__ not in versions:
        warnings.warn(
            f"""
            ultralytics {ultralytics.__version__} has not been tested.
            We recommend you to use one of {versions}
            """
        )
except ModuleNotFoundError as e:
    versions = get_backends()["ultralytics"]

    strings = [f"- pip install ultralytics=={version}" for version in versions]

    e.msg = "Need to install ultralytics\n" + "\n".join(strings)
    raise e


from . import BaseHub


class UltralyticsHub(BaseHub):

    AVAILABLE_TASK = ["detection", "classification", "segmentation"]

    def train(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

    def evaluation(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
