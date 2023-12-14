# it will be deleted after waffle_dough is released
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from waffle_utils.file.io import make_directory

Mat = np.ndarray[int, np.dtype[np.generic]]


def save_image(output_path: Union[str, Path], image: Mat, create_directory: bool = False) -> None:
    output_path = Path(output_path)
    if create_directory:
        make_directory(output_path.parent)

    save_type = output_path.suffix
    ret, img_arr = cv2.imencode(save_type, image)
    if ret:
        with open(str(output_path), mode="w+b") as f:
            img_arr.tofile(f)


def load_image(input_path: Union[str, Path]) -> Mat:
    return cv2.imdecode(np.fromfile(str(input_path), dtype=np.uint8), cv2.IMREAD_COLOR)
