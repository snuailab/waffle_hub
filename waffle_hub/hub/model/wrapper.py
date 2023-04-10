"""_summary_

Raises:
    NotImplementedError: _description_
    ValueError: _description_

Returns:
    _type_: _description_
"""

from typing import Union

import torch
from torchvision.ops import batched_nms

from waffle_hub.schema.data import (
    ClassificationResult,
    ImageInfo,
    ObjectDetectionResult,
)


class PreprocessFunction:
    pass


class PostprocessFunction:
    pass


class ResultParser:
    pass


class ClassificationResultParser(ResultParser):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(
        self,
        results: list[torch.Tensor],
        image_infos: list[ImageInfo] = None,
        *args,
        **kwargs
    ) -> list[ClassificationResult]:
        parseds = []

        results = results[0]  # TODO: multi label
        scores, class_ids = results.cpu().topk(results.shape[1], dim=-1)
        results = torch.cat(
            [class_ids.unsqueeze(-1), scores.unsqueeze(-1)], dim=-1
        )
        for result in results:
            parsed = []
            for class_id, score in result:
                parsed.append(
                    ClassificationResult(
                        category_id=int(class_id), score=float(score)
                    )
                )
            parseds.append(parsed)
        return parseds


class ObjectDetectionResultParser(ResultParser):
    def __init__(
        self,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        *args,
        **kwargs
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        results: list[torch.Tensor],
        image_infos: list[ImageInfo],
        *args,
        **kwargs
    ) -> list[ObjectDetectionResult]:
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
            for (x1, y1, x2, y2), conf, class_id in zip(
                bboxes, confs, class_ids
            ):

                x1 = max(float((x1 * W - left_pad) / new_w * ori_w), 0)
                y1 = max(float((y1 * H - top_pad) / new_h * ori_h), 0)
                x2 = min(float((x2 * W - left_pad) / new_w * ori_w), ori_w)
                y2 = min(float((y2 * H - top_pad) / new_h * ori_h), ori_h)

                parsed.append(
                    ObjectDetectionResult(
                        bbox=[x1, y1, x2 - x1, y2 - y1],
                        area=float((x2 - x1) * (y2 - y1)),
                        category_id=int(class_id),
                        score=float(conf),
                    )
                )
            parseds.append(parsed)
        return parseds


def get_parser(task: str):
    if task == "classification":
        return ClassificationResultParser
    elif task == "object_detection":
        return ObjectDetectionResultParser


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        preprocess: PreprocessFunction,
        postprocess: PostprocessFunction,
    ):
        """
        Model Wrapper.
        Use this wrapper when inference, export.

        Args:
            model (torch.nn.Module): model
            preprocess (PreprocessFunction):
                Preprocess Function that
                recieves [batch, channel, height, width] (0~1),
                and
                outputs [batch, channel, height, width].

            postprocess (PostprocessFunction):
                Postprocess Function that
                recieves model raw output,
                and
                outputs results that fit with our convention.

                Classification:
                    [
                        [batch, class_num],
                    ]  # scores per attribute
                Detection:
                    [
                        [batch, bbox_num, 4(x1, y1, x2, y2)],  # bounding box
                        [batch, bbox_num],  # confidence
                        [batch, bbox_num],  # class id
                    ]
                Segmentation:
                    # TODO: segmentation support
        """
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.preprocess(x)
        x = self.model(x)
        x = self.postprocess(x, image_size=(W, H))
        return x

    def get_layer_names(self) -> list[str]:
        """
        Get all layer names in model.
        """
        return [name for name, _ in self.model.named_modules()]

    def get_feature_maps(
        self, x, layer_names: Union[list[str], str] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Get feature maps from model.

        Args:
            x (torch.Tensor): input image
            layer_names (Union[list[str], str]): layer names to get feature maps

        Returns:
            x (torch.Tensor): model output
            feature_maps (dict): feature maps
        """

        feature_maps = {}

        def hook(name):
            def hook_fn(m, i, o):
                feature_maps[name] = o

            return hook_fn

        if layer_names is None:
            layer_names = self.get_layer_names()[-1]
        elif isinstance(layer_names, str):
            layer_names = [layer_names]

        for name, module in self.model.named_modules():
            if name in layer_names:
                print(name)
                module.register_forward_hook(hook(name))

        x = self.forward(x)

        return x, feature_maps
