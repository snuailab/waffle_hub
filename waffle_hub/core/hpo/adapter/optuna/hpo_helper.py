import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from waffle_utils.utils import type_validator

from .config import SEARCH_SPACE_MAP


class ChoiceMethod:
    def __init__(
        self, choice_name: str, choices: list, method_name: str = "suggest_float", **kwargs
    ):
        self.method_name = method_name
        self.choice_name = choice_name
        self.choices = choices
        self.kwargs = kwargs

    def __repr__(self):
        return f"{self.method_name}({self.choices}, {self.kwargs})"

    @property
    def method_name(self):
        return self.__method_name

    @method_name.setter
    @type_validator(str)
    def method_name(self, v):
        if v not in SEARCH_SPACE_MAP.keys():
            raise ValueError(f"method_name must be one of {list(SEARCH_SPACE_MAP.keys())}")
        self.__method_name = v

    @property
    def choice_name(self):
        return self.__choice_name

    @choice_name.setter
    @type_validator(str)
    def choice_name(self, v):
        self.__choice_name = v

    @property
    def choices(self):
        return self.__choices

    @choices.setter
    @type_validator(list)
    def choices(self, v):
        self.__choices = v

    def __call__(self, trial):
        if self.method_name == "suggest_categorical":
            return getattr(trial, self.method_name)(self.choice_name, self.choices)
        return getattr(trial, self.method_name)(self.choice_name, *self.choices)


def draw_error_image(message: str, image_path: str):
    # Create a white background image
    img = np.zeros((500, 700, 3), dtype=np.uint8)
    img.fill(255)

    # Convert the image to PIL format and draw the error message
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((0, 0), message, font_size=20, fill=(0))

    # Convert the image back to OpenCV format and save it to file
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, img)
