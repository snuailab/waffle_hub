import torch
from torch import nn

from waffle_hub.hub.model.wrapper import ModelWrapper
from waffle_hub.type import TaskType


class BoringModelWrapper(ModelWrapper):
    """Testing model wrapper.
    .. warning::  This is meant for testing/debugging and is experimental. ****

    Outputs:
        random tensor per task result

        CLASSIFICATION:
            [
                [batch, class_num],
            ]  # scores per attribute
        OBJECT_DETECTION:
            [
                [batch, bbox_num, 4(x1, y1, x2, y2)],  # bounding box
                [batch, bbox_num],  # confidence
                [batch, bbox_num],  # class id
            ]
        INSTANCE_SEGMENTATION:
            [
                [batch, pred_num, 4(x1, y1, x2, y2)],  # bounding box
                [batch, pred_num],  # confidence
                [batch, pred_num],  # class id
                [batch, pred_num, mask(H, W)] # warning: mask size and image size are not same
            ]
        SEMANTIC_SEGMENTATION:
            [
                [class_num, height, width]  # mask
            }
    """

    def __init__(self, task, categories, batch_size):
        model = nn.Sequential()
        _preprocess = lambda x: x
        if task == TaskType.CLASSIFICATION:

            def _postprocess(x: torch.Tensor, *args, **kwargs):
                return [torch.rand(batch_size, len(categories))]

        elif task == TaskType.OBJECT_DETECTION:

            def _postprocess(x: torch.Tensor, image_size: tuple[int, int], *args, **kwargs):
                xyxy = torch.rand(batch_size, 100, 4)
                confidences = torch.rand(batch_size, 100)
                class_ids = torch.rand(batch_size, 100)
                return xyxy, confidences, class_ids

        elif task == TaskType.INSTANCE_SEGMENTATION:

            def _postprocess(x: torch.Tensor, image_size: tuple[int, int], *args, **kwargs):
                xyxy = torch.rand(batch_size, 100, 4)
                confidences = torch.rand(batch_size, 100)
                class_ids = torch.rand(batch_size, 100)
                dummy_size = 20  # hard code
                masks = torch.rand(batch_size, 100, dummy_size, dummy_size)
                return xyxy, confidences, class_ids, masks

        elif task == TaskType.SEMANTIC_SEGMENTATION:

            def _postprocess(x: torch.Tensor, image_size: tuple[int, int], *args, **kwargs):
                masks = [
                    torch.rand(len(categories) + 1, image_size[0], image_size[1])
                    for _ in range(batch_size)
                ]
                return masks

        super().__init__(
            model=model,
            preprocess=_preprocess,
            postprocess=_postprocess,
            task=task,
            categories=categories,
        )
