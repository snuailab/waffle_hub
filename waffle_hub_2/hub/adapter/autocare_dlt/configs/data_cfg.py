def get_data_config(
    data_type: str,
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
    config = {
        "data": {
            "workers_per_gpu": workers,
            "batch_size_per_gpu": batch_size,
            "img_size": image_size,
            "train": {
                "type": data_type,
                "data_root": train_root,
                "ann": train_coco,
                "augmentation": {
                    "ColorJitter": {
                        "brightness": 0.125,
                        "contrast": 0.5,
                        "saturation": 0.5,
                        "hue": 0.1,
                    },
                    "Affine": {
                        "scale": [0.5, 1.5],
                        "translate_percent": [-0.1, 0.1],
                        "always_apply": True,
                    },
                    "HorizontalFlip": {"p": 0.5},
                    "Cutout": {
                        "p": 0.5,
                        "num_holes": 8,
                        "max_h_size": 32,
                        "max_w_size": 32,
                        "fill_value": 0,
                    },
                    "ImageNormalization": {"type": "base"},
                },
            },
            "val": {
                "type": data_type,
                "data_root": val_root,
                "ann": val_coco,
                "augmentation": {"ImageNormalization": {"type": "base"}},
            },
            "test": {
                "type": data_type,
                "data_root": test_root,
                "ann": test_coco,
                "augmentation": {"ImageNormalization": {"type": "base"}},
            },
            "cache": False,
            "single_cls": False,
        }
    }

    return config
