"""
Tx Model Hub
"""

from waffle_hub import get_installed_backend_version

BACKEND_NAME = "tx_model"
BACKEND_VERSION = get_installed_backend_version(BACKEND_NAME)

from dataclasses import asdict
from pathlib import Path
from typing import Union

import torch
from torchvision import transforms as T
from ultralytics import YOLO
from waffle_utils.file import io

from waffle_hub.utils.image import ImageDataset

from ..base_hub import BaseHub, InferenceContext, TrainContext
from ..model.wrapper import ModelWrapper, ResultParser, get_parser

def get_dataset_config(
    task: str, 
    image_size: list[int], 
    batch_size: int, 
    workers: int,
    train_coco: str,
    train_root: str,
    val_coco: str,
    val_root: str,
    test_coco: str,
    test_root: str,
):
    return {
        "data": {
            "workers_per_gpu": 4,
            "batch_size_per_gpu": 256,
            "img_size": [240, 80],
            "mode": "lpr",     
            "train": {
                "type": "COCOTextRecognitionDataset",
                "data_root": "data/DYIOT-licenseNum_v2/images",
                "ann": "LPRNet_v1.1.0/LPRNet_v1.1.0_ann_train.json",
                "augmentation": {
                    "ColorJitter":{
                        "brightness": 0.125,
                        "contrast": 0.5,
                        "saturation": 0.5,
                        "hue": 0.1
                    },
                    "Affine": {
                        "scale": [0.5, 1.1],
                        "translate_percent": [-0.1, 0.1], 
                        "always_apply": True
                    },
                    "Cutout": {
                        "p": 0.5,
                        "num_holes": 4,
                        "max_h_size": 8,
                        "max_w_size": 8,
                        "fill_value": 0
                    },
                    "ImageNormalization": {
                        "type": "base"
                    },
                    "SafeRotate": {
                        "limit": 30,
                        "p":0.5
                    }
                }
            },
            "val": {
                "type": "COCOTextRecognitionDataset",
                "data_root": "data/DYIOT-licenseNum_v2/images",
                "ann": "LPRNet_v1.1.0/LPRNet_v1.1.0_ann_val.json",
                "augmentation": {
                    "ImageNormalization": {
                        "type": "base"
                    }
                }
            },
            "test": {
                "type": "COCOTextRecognitionDataset",
                "data_root": "data/DYIOT-licenseNum_v2/images",
                "ann": "LPRNet_v1.1.0/LPRNet_v1.1.0_ann_val.json",
                "augmentation": {
                    "ImageNormalization": {
                        "type": "base"
                    }
                }
            },
            "cache": False,
            "single_cls": False
        }
    } 


class TxModelHub(BaseHub):

    # Common
    MODEL_TYPES = {
        "object_detection": {
            "yolov5": list("sml")
        },
        "classification": {
            "resnet": list("sml"),
            "swin": list("sml")
        },
    }

    # Backend Specifics
    TASK_MAP = {
        "object_detection": "detect",
        "classification": "classify",
        # "segmentation": "segment"
        # "keypoint_detection": "pose"
    }