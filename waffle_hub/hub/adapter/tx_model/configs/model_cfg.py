def get_model_config(
    model_type: str,
    model_size: str,
    categories: list[str],
    seed: int,
    letter_box: bool,
    epochs: int,
):
    return {
        "task": model_type,
        "model": {
            "model_size": model_size,
            "backbone": {"name": "YOLOv5Backbone", "focus": False},
            "neck": {"name": "YOLOv5Neck"},
            "head": {
                "name": "YOLOv5Head",
                "anchors": [
                    [10, 13, 16, 30, 33, 23],
                    [30, 61, 62, 45, 59, 119],
                    [116, 90, 156, 198, 373, 326],
                ],
            },
        },
        "loss": {
            "total_loss": {
                "name": "YoloLoss",
                "params": {
                    "hyp": {
                        "box": 0.05,
                        "cls": 0.3,
                        "cls_pw": 1.0,
                        "obj": 0.7,
                        "obj_pw": 1.0,
                        "anchor_t": 4.0,
                        "fl_alpha": -1,
                        "fl_gamma": 0.0,
                        "label_smoothing": 0.0,
                    }
                },
            }
        },
        "optim": {
            "name": "SGD",
            "lr": 0.001,
            "momentum": 0.937,
            "weight_decay": 0.0005,
        },
        "lr_cfg": {
            "type": "cosine",
            "warmup": True,
            "warmup_epochs": 0.5,
            "lrf": 0.1,
        },
        "ema_cfg": {"burn_in_epoch": 1},
        "max_epoch": epochs,
        "nms_thresh": 0.65,
        "min_score": 0.25,
        "detections_per_img": 300,
        "seed": seed,
        "categories": categories,
        "num_categories": len(categories),
        "letter_box": letter_box,
    }
