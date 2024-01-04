from typing import Any

import cv2
from torch.utils.data import DataLoader

from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.inferencer.callbacks import BaseInferenceCallback
from waffle_hub.hub.inferencer.inferencer import Inferencer
from waffle_hub.utils.draw import draw_results


class InferenceShowCallback(BaseInferenceCallback):
    def __init__(self):
        pass

    def teardown(self, inferencer: Inferencer) -> None:
        """Called when worker ends."""
        cv2.destroyAllWindows()

    def on_inference_step_end(
        self, inferencer: Inferencer, step: int, batch: Any, result_batch: Any
    ) -> None:
        """Called when the inference loop step ends."""
        images, image_infos = batch
        for result, image_info in zip(result_batch, image_infos):
            draw = draw_results(
                image_info.ori_image,
                result,
                names=[x["name"] for x in inferencer.model.categories],
            )
            cv2.imshow("result", draw)
            cv2.waitKey(1)
