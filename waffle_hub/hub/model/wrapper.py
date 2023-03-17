"""_summary_

Raises:
    NotImplementedError: _description_
    ValueError: _description_

Returns:
    _type_: _description_
"""

import torch


class PreprocessFunction:
    pass


class PostprocessFunction:
    pass


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
                recieves [batch, channel, height, width],
                and
                outputs [batch, channel, height, width].

            postprocess (PostprocessFunction):
                Postprocess Function that
                recieves model raw output,
                and
                outputs results that fit with our convention.

                Classification:
                    [batch, class_num]  # scores
                Detection:
                    [
                        [batch, bbox_num, 4(x1, y1, x2, y2)],  # bounding box
                        [batch, bbox_num],  # confidence
                        [batch, bbox_num],  # class id
                    ]
                Segmentation:
                    # TODO
        """
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)
        x = self.postprocess(x)
        return x


def get_result_parser(task: str, *args, **kwargs):
    """Get Result Parser

    Args:
        task (str): task name.
    """

    if task == "classification":

        def parser(results: torch.Tensor, image_info: dict):
            scores, class_ids = results.cpu().topk(results.shape[1], dim=-1)
            return torch.cat(
                [class_ids.unsqueeze(-1), scores.unsqueeze(-1)], dim=-1
            )

    elif task == "object_detection":

        def parser(results, image_info):
            pass

    elif task == "segmentation":
        raise NotImplementedError(f"{task} is not supported yet.")

    else:
        raise ValueError(f"Unknow task: {task}")

    return parser
