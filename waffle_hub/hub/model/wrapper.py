"""_summary_

Raises:
    NotImplementedError: _description_
    ValueError: _description_

Returns:
    _type_: _description_
"""

import torch
from torchvision.ops import batched_nms


class PreprocessFunction:
    pass


class PostprocessFunction:
    pass


class ResultParser:
    pass


class ClassificationResultParser(ResultParser):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, results: list[torch.Tensor], *args, **kwargs):
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
                    {"category_id": int(class_id), "score": float(score)}
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
        image_infos: list[dict],
        *args,
        **kwargs
    ):
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

            W, H = image_info.get("input_shape")
            left_pad, top_pad = image_info.get("pad")
            ori_w, ori_h = image_info.get("ori_shape")
            new_w, new_h = image_info.get("new_shape")

            parsed = []
            for (x1, y1, x2, y2), conf, class_id in zip(
                bboxes, confs, class_ids
            ):

                x1 = max(float((x1 * W - left_pad) / new_w * ori_w), 0)
                y1 = max(float((y1 * H - top_pad) / new_h * ori_h), 0)
                x2 = min(float((x2 * W - left_pad) / new_w * ori_w), ori_w)
                y2 = min(float((y2 * H - top_pad) / new_h * ori_h), ori_h)

                parsed.append(
                    {
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "category_id": int(class_id),
                        "score": float(conf),
                    }
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
        parser: ResultParser = None,
    ):
        """
        Model Wrapper.
        Use this wrapper when inference, export.

        Args:
            model (torch.nn.Module): model
            preprocess (PreprocessFunction):
                Preprocess Function that
                recieves [batch, channel, height, width],
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
        self.parser = parser

    def forward(self, x, image_infos: list[dict] = None, *args, **kwargs):
        x = self.preprocess(x)
        x = self.model(x)
        x = self.postprocess(x)
        if self.parser is not None:
            x = self.parser(x, image_infos=image_infos, *args, **kwargs)
        return x
