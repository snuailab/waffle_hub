from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torchvision import transforms as T
from waffle_utils.file import io

from waffle_hub.dataset import Dataset
from waffle_hub.schema.data import ImageInfo
from waffle_hub.schema.fields import Annotation, Category, Image


def get_images(d, recursive: bool = True) -> list[str]:
    exp = "**/*" if recursive else "*"
    image_paths = []
    for ext in ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]:
        image_paths += list(Path(d).glob(exp.lower() + "." + ext))
        image_paths += list(Path(d).glob(exp.upper() + "." + ext))
    return list(
        set(
            map(
                str,
                image_paths,
            )
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
        image = cv2.copyMakeBorder(image, top, bottom, left, right, None, value=(114, 114, 114))

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


def get_image_transform(image_size: Union[int, list[int]], letter_box: bool = False):

    if isinstance(image_size, int):
        image_size = [image_size, image_size]

    def transform(image: Union[np.ndarray, str]) -> tuple[torch.Tensor, ImageInfo]:
        if isinstance(image, str):
            image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, image_info = resize_image(image, image_size, letter_box)
        return T.ToTensor()(image), image_info

    return transform


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

        self.transform = get_image_transform(image_size, letter_box)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image, image_info = self.transform(image_path)
        image_info.image_path = image_path

        return image, image_info

    def collate_fn(self, batch):
        images, infos = list(zip(*batch))
        return torch.stack(images, dim=0), infos

    def get_dataloader(self, batch_size: int = 4, num_workers: int = 0):
        return torch.utils.data.DataLoader(
            self,
            batch_size,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=False,
        )


class LabeledDataset:
    def __init__(
        self,
        dataset: Dataset,
        image_size: Union[int, list[int]],
        letter_box: bool = False,
        set_name: str = None,
    ):
        self.dataset = dataset
        self.image_dir = dataset.raw_image_dir
        self.set_name = set_name

        if self.set_name == "train":
            set_file = self.dataset.train_set_file
        elif self.set_name == "val":
            set_file = self.dataset.val_set_file
        elif self.set_name == "test":
            set_file = self.dataset.test_set_file
        else:
            set_file = None

        self.images: list[Image] = self.dataset.get_images(
            io.load_json(set_file) if set_file else None
        )
        self.image_to_annotations: dict[int, list[Annotation]] = {
            image.image_id: self.dataset.get_annotations(image.image_id) for image in self.images
        }

        self.transform = get_image_transform(image_size, letter_box)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_path = str(self.image_dir / image.file_name)
        annotations: list[Annotation] = self.image_to_annotations[image.image_id]

        image, image_info = self.transform(image_path)
        image_info.image_path = image_path

        return image, image_info, annotations

    def collate_fn(self, batch):
        images, infos, annotations = list(zip(*batch))
        return torch.stack(images, dim=0), infos, annotations

    def get_dataloader(self, batch_size: int = 4, num_workers: int = 0):
        return torch.utils.data.DataLoader(
            self,
            batch_size,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=False,
        )
