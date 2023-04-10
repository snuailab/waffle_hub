from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torchvision import transforms as T

from waffle_hub.schema.data import ImageInfo


def get_images(d, recursive: bool = True) -> list[str]:
    exp = "**/*" if recursive else "*"
    return list(
        map(
            str,
            list(Path(d).glob(exp + ".png"))
            + list(Path(d).glob(exp + ".jpg"))
            + list(Path(d).glob(exp + ".PNG"))
            + list(Path(d).glob(exp + ".JPG")),
        )
    )


def resize_image(
    image: np.ndarray, image_size: list[int], letter_box: bool = False
) -> list[np.ndarray, ImageInfo]:
    """Resize Image.

    Args:
        image (np.ndarray): opencv image.
        image_size (list[int]): image [width, height].
        letter_box (bool): letter box.

    Returns:
        list[np.ndarray, ImageInfo]: resized image, image info.
    """

    h, w = image.shape[:2]
    W, H = image_size

    if letter_box:
        if w > h:
            ratio = W / w
            w_ = W
            h_ = round(h * ratio)

            total_pad = w_ - h_
            top = total_pad // 2
            bottom = total_pad - top
            left, right = 0, 0

        else:
            ratio = H / h
            w_ = round(w * ratio)
            h_ = H

            total_pad = h_ - w_
            left = total_pad // 2
            right = total_pad - left
            top, bottom = 0, 0

        image = cv2.resize(image, (w_, h_), interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, None, value=(114, 114, 114)
        )

    else:
        w_, h_ = W, H
        left, top = 0, 0
        image = cv2.resize(image, (w_, h_), interpolation=cv2.INTER_CUBIC)

    return image, ImageInfo(
        ori_shape=(w, h),
        new_shape=(w_, h_),
        input_shape=(W, H),
        pad=(left, top),
    )


def collate_fn(batch):
    images, infos = list(zip(*batch))
    return torch.stack(images, dim=0), infos


class ImageDataset:
    def __init__(
        self,
        image_dir: str,
        image_size: Union[int, list[int]],
        letter_box: bool = False,
    ):
        self.image_dir = image_dir
        if Path(self.image_dir).is_file():
            self.image_paths = [self.image_dir]
        else:
            self.image_paths = get_images(self.image_dir)

        self.image_size = (
            image_size
            if isinstance(image_size, list)
            else [image_size, image_size]
        )
        self.letter_box = letter_box

        self.transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, image_info = resize_image(
            image, self.image_size, self.letter_box
        )

        image_info.image_path = image_path

        return self.transform(image), image_info

    def get_dataloader(self, batch_size: int, num_workers: int):
        return torch.utils.data.DataLoader(
            self,
            batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
        )
