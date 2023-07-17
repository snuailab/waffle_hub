import warnings
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from natsort import natsorted
from torchvision import transforms as T
from waffle_utils.file import io

from waffle_hub.dataset import Dataset
from waffle_hub.schema.data import ImageInfo
from waffle_hub.schema.fields import Annotation, Category, Image

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]


def get_images(d, recursive: bool = True) -> list[str]:
    exp = "**/*" if recursive else "*"
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths += list(Path(d).glob(exp.lower() + ext))
        image_paths += list(Path(d).glob(exp.upper() + ext))
    return natsorted(
        list(
            set(
                map(
                    str,
                    image_paths,
                )
            )
        )
    )


def get_videos(d, recursive: bool = True) -> list[str]:
    exp = "**/*" if recursive else "*"
    video_paths = []
    for ext in VIDEO_EXTS:
        video_paths += list(Path(d).glob(exp.lower() + ext))
        video_paths += list(Path(d).glob(exp.upper() + ext))
    return natsorted(
        list(
            set(
                map(
                    str,
                    video_paths,
                )
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

        resized_image = cv2.resize(image, (w_, h_), interpolation=cv2.INTER_CUBIC)
        resized_image = cv2.copyMakeBorder(
            resized_image, top, bottom, left, right, None, value=(114, 114, 114)
        )

    else:
        w_, h_ = W, H
        left, top = 0, 0
        resized_image = cv2.resize(image, (w_, h_), interpolation=cv2.INTER_CUBIC)

    return resized_image, ImageInfo(
        ori_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        ori_shape=(w, h),
        new_shape=(w_, h_),
        input_shape=(W, H),
        pad=(left, top),
    )


def get_image_transform(image_size: Union[int, list[int]], letter_box: bool = False):
    def transform(image: Union[np.ndarray, str]) -> tuple[torch.Tensor, ImageInfo]:
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, image_info = resize_image(image, image_size, letter_box)
        return T.ToTensor()(image), image_info

    return transform


def get_dataset_class(dataset_type: str):
    if dataset_type == "image":
        return ImageDataset
    elif dataset_type == "video":
        return VideoDataset
    elif dataset_type == "dataset":
        return LabeledDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


class BaseDataset:
    def __init__(
        self,
        image_size: Union[int, list[int]],
        letter_box: bool = False,
    ):
        if isinstance(image_size, int):
            image_size = [image_size, image_size]

        if len(image_size) != 2:
            raise ValueError("image_size must be a int or list of int with length 2(width, height).")

        self.image_size = image_size
        self.letter_box = letter_box

        self.transform = get_image_transform(self.image_size, self.letter_box)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def collate_fn(self, batch):
        raise NotImplementedError

    def get_dataloader(self, batch_size: int = 4, num_workers: int = 0):
        return torch.utils.data.DataLoader(
            self,
            batch_size,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=False,
        )


class ImageDataset(BaseDataset):
    def __init__(
        self,
        image_dir: str,
        image_size: Union[int, list[int]],
        letter_box: bool = False,
        recursive: bool = True,
        **kwargs,
    ):
        super().__init__(image_size, letter_box)

        self.image_dir = image_dir
        if Path(self.image_dir).is_file():
            self.image_paths = [self.image_dir]
            self.image_root_dir = Path(self.image_dir).parent
        else:
            self.image_paths = get_images(self.image_dir, recursive=recursive)
            self.image_root_dir = Path(self.image_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image_tensor, image_info = self.transform(image_path)
        image_info.image_path = image_path
        image_info.image_rel_path = str(Path(image_path).relative_to(self.image_root_dir))

        return image_tensor, image_info

    def collate_fn(self, batch):
        images, infos = list(zip(*batch))
        return torch.stack(images, dim=0), infos


class LabeledDataset(BaseDataset):
    def __init__(
        self,
        dataset: Dataset,
        image_size: Union[int, list[int]],
        letter_box: bool = False,
        set_name: str = None,
        **kwargs,
    ):
        super().__init__(image_size, letter_box)

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_path = str(self.image_dir / image.file_name)
        annotations: list[Annotation] = self.image_to_annotations[image.image_id]

        image_tensor, image_info = self.transform(image_path)
        image_info.image_path = image_path
        image_info.image_rel_path = image.file_name

        return image_tensor, image_info, annotations

    def collate_fn(self, batch):
        images, infos, annotations = list(zip(*batch))
        return torch.stack(images, dim=0), infos, annotations


class VideoDataset(BaseDataset):
    def __init__(
        self,
        video_path: str,
        image_size: Union[int, list[int]],
        letter_box: bool = False,
        **kwargs,
    ):
        super().__init__(image_size, letter_box)

        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {idx} from video {self.video_path}.")
        image_tensor, image_info = self.transform(frame)
        image_info.image_rel_path = f"{idx}.png"

        return image_tensor, image_info

    def collate_fn(self, batch):
        images, infos = list(zip(*batch))
        return torch.stack(images, dim=0), infos

    def get_dataloader(self, batch_size: int = 1, num_workers: int = 0):
        if batch_size > 1:
            warnings.warn("batch_size > 1 is not supported for video dataset.")
        if num_workers > 0:
            warnings.warn("num_workers > 0 is not supported for video dataset.")
        return super().get_dataloader(batch_size=1, num_workers=0)
