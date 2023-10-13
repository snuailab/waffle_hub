import os
import warnings
from pathlib import Path

from waffle_utils.file import io, network


class A:
    def __init__(self, num1):
        self.num1 = num1

    def sim(self, num2):
        if self.num1 == 0:
            raise ValueError("num1 cannot be 0")
        return self.num1 * num2


def yolo_object_detection_path(path_data=Path("/home/daeun/workspace/waffle_hub/datasets")):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_yolo_object_detection.zip"

    zip_file = path_data / "mnist_yolo_object_detection.zip"
    yolo_path = path_data / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, yolo_path, create_directory=True)

    return Path(yolo_path)


if __name__ == "__main__":
    yolo_object_detection_path()
