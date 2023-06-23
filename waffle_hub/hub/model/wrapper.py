"""_summary_

Raises:
    NotImplementedError: _description_
    ValueError: _description_

Returns:
    _type_: _description_
"""

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from waffle_hub import TaskType
from waffle_hub.schema.data import ImageInfo
from waffle_hub.schema.fields import Annotation
from waffle_hub.utils.conversion import convert_mask_to_polygon


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


def get_parser(task: str):
    if task == TaskType.CLASSIFICATION:
        return ClassificationResultParser
    elif task == TaskType.OBJECT_DETECTION:
        return ObjectDetectionResultParser
    elif task == TaskType.INSTANCE_SEGMENTATION:
        return InstanceSegmentationResultParser
    elif task == TaskType.TEXT_RECOGNITION:
        return TextRecognitionResultParser


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
                    [
                        [batch, bbox_num, 4(x1, y1, x2, y2)],  # bounding box
                        [batch, bbox_num],  # confidence
                        [batch, bbox_num],  # class id
                        [batch, mask(H, W)] # warning: mask size and image size are not same
                    ]
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

    def get_layers(self, layer_names: Union[list[str], str]) -> list[torch.nn.Module]:
        """
        Get layer in model by name.

        Args:
            layer_names (Union[list[str], str]): layer names to get

        Returns:
            layers (list[torch.nn.Module]): layers
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        return [layer for name, layer in self.model.named_modules() if name in layer_names]

    def _convert_to_feature_map(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Convert feature to feature map.
        """
        if len(feature.shape) == 4:  # Convolution Feature Map
            return feature
        elif len(feature.shape) == 3:  # Linear(ViT) Feature Map
            batch, dim, channel = feature.shape
            size = int(np.sqrt(dim - 1))
            return feature[:, 1:].view(
                batch, channel, size, size
            )  # TODO: first token is cls token (usually)
        else:
            raise ValueError(f"Unsupported feature map type. {feature.shape}")

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
                module.register_forward_hook(hook(name))

        x = self.forward(x)

        return x, feature_maps

    def get_cam(
        self,
        x: torch.Tensor,
        layer_name: str,
        class_id: int = None,
    ) -> torch.Tensor:
        """
        TODO: experimental
        Get class activation map.

        Args:
            x (torch.Tensor): input image
            layer_name (str): layer name to get feature maps
            class_id (int): class id to get cam
            image_size (tuple[int, int]): image size
            upsample_size (tuple[int, int]): upsample size

        Returns:
            cam (torch.Tensor): class activation map
        """

        activation_list = []
        gradient_list = []

        def hook(name):
            def hook_fn(m, i, o):
                activation_list.append(self._convert_to_feature_map(o).cpu().detach())

                if not hasattr(o, "requires_grad" or not o.requires_grad):
                    return

                def _store_grad(grad):
                    gradient_list.insert(0, self._convert_to_feature_map(grad).cpu().detach())

                o.register_hook(_store_grad)

            return hook_fn

        layer = self.get_layers(layer_name)[0]
        layer.requires_grad_(True)
        handle = layer.register_forward_hook(hook(layer_name))

        self.model.zero_grad()
        output = self.forward(x)[0]
        if class_id is None:
            class_id = torch.argmax(output[0])
        output[:, class_id].backward()

        activations = activation_list[0]
        gradients = gradient_list[0]

        if activations.shape != gradients.shape:
            raise ValueError(
                f"Activation shape {activations.shape} and gradient shape {gradients.shape} are not same."
            )

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # release hooks
        handle.remove()

        # normalize cam to (0, 1)
        cam = torch.nn.functional.relu(cam)
        cam_max = torch.max(cam)
        cam_min = torch.min(cam)
        cam = (cam - cam_min) / (cam_max - cam_min)

        # resize cam
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return cam
