from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torchvision import transforms as T


# TODO: draw utils..
def draw_results(
    image: Union[np.ndarray, str],
    results: list[dict],
    task: str,
    names: list[str],
):

    if task == "classification":

        def draw(image, results):
            image = cv2.putText(
                image,
                f"{names[results[0].get('category_id')]}",
                (10, 30),
                2,
                1.0,
                (0, 0, 255),
            )
            return image

    elif task == "object_detection":

        def draw(image, results):
            for result in results:
                x1, y1, w, h = result["bbox"]
                x2 = x1 + w
                y2 = y1 + h
                image = cv2.putText(
                    image,
                    f"{names[result.get('category_id')]}",
                    (int(x1), int(y1) - 2),
                    2,
                    1.0,
                    (0, 0, 255),
                )
                image = cv2.rectangle(
                    image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    2,
                )
            return image

    else:
        raise NotImplementedError(f"{task} draw function not implemented")

    if isinstance(image, str):
        image = cv2.imread(image)

    return draw(image, results)


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


# TODO: define image_info schema
def resize_image(
    image: np.ndarray, image_size: list[int], letter_box: bool = False
) -> list[np.ndarray, list[int], list[int], list[int], list[int]]:
    """Resize Image.

    Args:
        image (np.ndarray): opencv image.
        image_size (list[int]): image [width, height].
        letter_box (bool): letter box.

    Returns:
        list: resized_image, (ori_width, ori_height), (new_width, new_height), (left_pad, top_pad)
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

        return image, (w, h), (w_, h_), (W, H), (left, top)

    else:
        w_, h_ = W, H
        image = cv2.resize(image, (w_, h_), interpolation=cv2.INTER_CUBIC)

        return image, (w, h), (w_, h_), (W, H), (0, 0)


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
        (
            image,
            (ori_w, ori_h),
            (new_w, new_h),
            (input_w, input_h),
            (left_pad, top_pad),
        ) = resize_image(image, self.image_size, self.letter_box)

        return self.transform(image), {
            "image_path": image_path,
            "ori_shape": (ori_w, ori_h),
            "new_shape": (new_w, new_h),
            "input_shape": (input_w, input_h),
            "pad": (left_pad, top_pad),
        }

    def get_dataloader(self, batch_size: int, num_workers: int):
        return torch.utils.data.DataLoader(
            self,
            batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
        )
