from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from waffle_hub.schema.data import ImageInfo
from waffle_hub.schema.fields import Annotation
from waffle_hub.type import TaskType
from waffle_hub.utils.conversion import convert_mask_to_polygon


class ResultParser:
    pass


class ClassificationResultParser(ResultParser):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(
        self, results: list[torch.Tensor], image_infos: list[ImageInfo] = None, *args, **kwargs
    ) -> list[Annotation]:
        parseds = []

        results = results[0]  # TODO: multi label
        scores, class_ids = results.cpu().topk(results.shape[1], dim=-1)
        results = torch.cat([class_ids.unsqueeze(-1), scores.unsqueeze(-1)], dim=-1)
        for result in results:
            parsed = []
            for class_id, score in result:
                parsed.append(
                    Annotation.classification(category_id=int(class_id) + 1, score=float(score))
                )
            parseds.append(parsed)
        return parseds


class ObjectDetectionResultParser(ResultParser):
    def __init__(
        self, confidence_threshold: float = 0.25, iou_threshold: float = 0.5, *args, **kwargs
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def __call__(
        self, results: list[torch.Tensor], image_infos: list[ImageInfo], *args, **kwargs
    ) -> list[Annotation]:
        parseds = []

        bboxes_batch, confs_batch, class_ids_batch = results
        for bboxes, confs, class_ids, image_info in zip(
            bboxes_batch, confs_batch, class_ids_batch, image_infos
        ):

            mask = confs > self.confidence_threshold

            bboxes, confs, class_ids = (
                bboxes[mask],
                confs[mask],
                class_ids[mask],
            )
            idxs = batched_nms(bboxes, confs, class_ids, self.iou_threshold)

            bboxes = bboxes[idxs, :].cpu()
            confs = confs[idxs].cpu()
            class_ids = class_ids[idxs].cpu()

            W, H = image_info.input_shape
            left_pad, top_pad = image_info.pad
            ori_w, ori_h = image_info.ori_shape
            new_w, new_h = image_info.new_shape

            parsed = []
            for (x1, y1, x2, y2), conf, class_id in zip(bboxes, confs, class_ids):

                x1 = max(float((x1 * W - left_pad) / new_w * ori_w), 0)
                y1 = max(float((y1 * H - top_pad) / new_h * ori_h), 0)
                x2 = min(float((x2 * W - left_pad) / new_w * ori_w), ori_w)
                y2 = min(float((y2 * H - top_pad) / new_h * ori_h), ori_h)

                parsed.append(
                    Annotation.object_detection(
                        category_id=int(class_id) + 1,
                        bbox=[x1, y1, x2 - x1, y2 - y1],
                        area=float((x2 - x1) * (y2 - y1)),
                        score=float(conf),
                    )
                )
            parseds.append(parsed)
        return parseds


class InstanceSegmentationResultParser(ObjectDetectionResultParser):
    def __init__(
        self, confidence_threshold: float = 0.25, iou_threshold: float = 0.5, *args, **kwargs
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        super().__init__(confidence_threshold, iou_threshold, *args, **kwargs)

    def __call__(
        self, results: list[torch.Tensor], image_infos: list[ImageInfo], *args, **kwargs
    ) -> list[Annotation]:
        parseds = []
        bboxes_batch, confs_batch, class_ids_batch, masks_batch = results
        for bboxes, confs, class_ids, masks, image_info in zip(
            bboxes_batch,
            confs_batch,
            class_ids_batch,
            masks_batch,
            image_infos,
        ):

            mask = confs > self.confidence_threshold

            bboxes, confs, class_ids, masks = (
                bboxes[mask],
                confs[mask],
                class_ids[mask],
                masks[mask],
            )
            idxs = batched_nms(bboxes, confs, class_ids, self.iou_threshold)

            bboxes = bboxes[idxs, :].cpu()
            confs = confs[idxs].cpu()
            class_ids = class_ids[idxs].cpu()

            masks = masks[idxs, :]
            masks = (
                F.interpolate(
                    input=masks.gt_(0.5).unsqueeze(1),
                    size=image_info.ori_shape,
                    mode="bilinear",
                    align_corners=False,
                )
                .cpu()
                .squeeze(1)
            )

            W, H = image_info.input_shape
            left_pad, top_pad = image_info.pad
            ori_w, ori_h = image_info.ori_shape
            new_w, new_h = image_info.new_shape

            parsed = []
            for (x1, y1, x2, y2), conf, class_id, mask in zip(bboxes, confs, class_ids, masks):

                x1 = max(float((x1 * W - left_pad) / new_w * ori_w), 0)
                y1 = max(float((y1 * H - top_pad) / new_h * ori_h), 0)
                x2 = min(float((x2 * W - left_pad) / new_w * ori_w), ori_w)
                y2 = min(float((y2 * H - top_pad) / new_h * ori_h), ori_h)

                # clean non roi area
                mask[:, : round(x1)] = 0
                mask[:, round(x2) :] = 0
                mask[: round(y1), :] = 0
                mask[round(y2) :, :] = 0

                segment = convert_mask_to_polygon(mask.numpy().astype(np.uint8))

                parsed.append(
                    Annotation.instance_segmentation(
                        category_id=int(class_id) + 1,
                        bbox=[x1, y1, x2 - x1, y2 - y1],
                        area=float((x2 - x1) * (y2 - y1)),
                        score=float(conf),
                        segmentation=segment,
                    )
                )
            parseds.append(parsed)
        return parseds


class TextRecognitionResultParser(ResultParser):
    def __init__(self, categories, *args, **kwargs):
        self.categories = categories
        self.category_names = [""] + [d["name"] for d in self.categories]

    def __call__(
        self, results: list[torch.Tensor], image_infos: list[ImageInfo] = None, *args, **kwargs
    ) -> list[Annotation]:
        parseds = []

        pred_batch, conf_batch = results
        for preds, confs in zip(pred_batch, conf_batch):
            mask = preds > 0
            preds, confs = preds[mask], confs[mask]

            text = "".join([self.category_names[pred] for pred in preds])

            parsed = []
            parsed.append(
                Annotation.text_recognition(
                    score=list(map(float, confs)),
                    caption=text,
                )
            )
            parseds.append(parsed)

        return parseds


class SemanticSegmentationResultParser(ResultParser):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(
        self, results: list[torch.Tensor], image_infos: list[ImageInfo], *args, **kwargs
    ) -> list[Annotation]:
        parseds = []

        for masks, image_info in zip(
            results,
            image_infos,
        ):
            parsed = []

            left_pad, top_pad = image_info.pad

            num_class = masks.shape[0]
            mask = masks.argmax(dim=0)
            class_masks = [(mask == class_id) * 255 for class_id in range(num_class - 1)]
            for class_id, class_mask in enumerate(class_masks):
                class_mask = class_mask.numpy().astype(np.uint8)
                if left_pad > 0:
                    class_mask = class_mask[:, left_pad:-left_pad]
                if top_pad > 0:
                    class_mask = class_mask[top_pad:-top_pad, :]

                class_mask = cv2.resize(class_mask, dsize=(image_info.ori_shape))
                segment = convert_mask_to_polygon(class_mask)

                if segment:
                    parsed.append(
                        Annotation.semantic_segmentation(
                            category_id=int(class_id) + 1,
                            segmentation=segment,
                        )
                    )

            parseds.append(parsed)
        return parseds


def get_parser(task: str) -> ResultParser:
    if task == TaskType.CLASSIFICATION:
        return ClassificationResultParser
    elif task == TaskType.OBJECT_DETECTION:
        return ObjectDetectionResultParser
    elif task == TaskType.INSTANCE_SEGMENTATION:
        return InstanceSegmentationResultParser
    elif task == TaskType.TEXT_RECOGNITION:
        return TextRecognitionResultParser
    elif task == TaskType.SEMANTIC_SEGMENTATION:
        return SemanticSegmentationResultParser
    else:
        raise ValueError(f"Unsupported task type: {task}")
