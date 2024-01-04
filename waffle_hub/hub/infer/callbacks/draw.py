from pathlib import Path
from typing import Any, Union

from torch.utils.data import DataLoader
from waffle_utils.file import io

from temp_utils.image.io import save_image
from temp_utils.video.io import create_video_writer
from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.infer.callbacks import BaseInferenceCallback
from waffle_hub.hub.infer.inferencer import Inferencer
from waffle_hub.utils.draw import draw_results


class InferenceDrawCallback(BaseInferenceCallback):
    def __init__(self, draw_dir: Union[Path, str]):
        self.draw_dir = Path(draw_dir)

    def setup(self, inferencer: Inferencer) -> None:
        """Called when worker starts."""
        self.video_writer = None

    def teardown(self, inferencer: Inferencer) -> None:
        """Called when worker ends."""
        if self.video_writer is not None:
            self.video_writer.release()

    def on_inference_loop_start(
        self, inferencer: Inferencer, dataset: Dataset, dataloader: DataLoader
    ) -> None:
        """Called when the inference loop begins."""
        io.make_directory(self.draw_dir)

        if inferencer.cfg.source_type == "video":  # for video writer
            self.dataset_fps = dataset.fps

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

            if inferencer.cfg.source_type == "video":
                if self.video_writer is None:
                    h, w = draw.shape[:2]
                    self.video_writer = create_video_writer(
                        str(
                            inferencer.inference_dir
                            / Path(inferencer.cfg.source).with_suffix(".mp4").name
                        ),
                        self.dataset_fps,
                        (w, h),
                    )
                self.video_writer.write(draw)

                draw_path = (
                    self.draw_dir
                    / Path(inferencer.cfg.source).stem
                    / Path(image_info.image_rel_path).with_suffix(".png")
                )
            else:
                draw_path = self.draw_dir / Path(image_info.image_rel_path).with_suffix(".png")
            save_image(draw_path, draw, create_directory=True)

    def after_inference(self, inferencer: Inferencer) -> None:
        """Called when the inference ends."""
        inferencer.result.draw_dir = self.draw_dir
