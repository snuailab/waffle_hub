import os
import warnings
from pathlib import Path

from waffle_utils.callback import BaseCallback

from waffle_hub.hub.train.adpater.ultralytics.hook import UltralyticsTrainHook
from waffle_hub.hub.train.trainer import BaseTrainer


class UltralyticsTrainer(BaseTrainer, UltralyticsTrainHook):
    # Train spec, abstract property
    MULTI_GPU_TRAIN = True

    def __init__(self, root_dir: Path, callbacks: list[BaseCallback] = None):
        BaseTrainer.__init__(self, root_dir=root_dir)
        UltralyticsTrainHook.__init__(self, callbacks=callbacks)

    def get_metrics(self) -> list[list[dict]]:
        # read csv file
        # epoch,         train/box_loss,         train/cls_loss,         train/dfl_loss,   metrics/precision(B),      metrics/recall(B),       metrics/mAP50(B),    metrics/mAP50-95(B),           val/box_loss,           val/cls_loss,           val/dfl_loss,                 lr/pg0,                 lr/pg1,                 lr/pg2
        #              0,                 2.0349,                 4.4739,                  1.214,                0.75962,                0.34442,                0.22858,                0.18438,                0.61141,                 2.8409,                0.78766,                 0.0919,                 0.0009,                 0.0009
        #              1,                 1.7328,                 4.0078,                 1.1147,                0.12267,                0.40909,                0.20891,                0.15007,                0.75082,                 2.7157,                0.79723,               0.082862,              0.0018624,              0.0018624
        # and parse it to list[dict]
        # [[{"tag": train/box_loss, "value": 0.0}, {"tag": train/box_loss, "value": 1.7328}, ...], ...]
        # and return it

        csv_path = self.artifacts_dir / "results.csv"

        if not csv_path.exists():
            return []

        with open(csv_path) as f:
            lines = f.readlines()

        header = lines[0].strip().split(",")
        metrics = []
        for line in lines[1:]:
            values = line.strip().split(",")
            metric = []
            for i, value in enumerate(values):
                metric.append(
                    {
                        "tag": header[i].strip(),
                        "value": float(value),
                    }
                )
            metrics.append(metric)

        return metrics
