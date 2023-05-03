# Common
MODEL_TYPES = {
    "object_detection": {"yolov8": list("nsmlx")},
    "classification": {"yolov8": list("nsmlx")},
    "instance_segmentation": {"yolov8": list("nsmlx")},
    # "keypoint_detection": {"yolov8": list("nsmlx")},
}

# Backend Specifics
TASK_MAP = {
    "object_detection": "detect",
    "classification": "classify",
    "instance_segmentation": "segment"
    # "keypoint_detection": "pose"
}
TASK_SUFFIX = {
    "detect": "",
    "classify": "-cls",
    "segment": "-seg",
}

DEFAULT_PARAMAS = {
    "object_detection": {
        "yolov8": {
            "n": {
                "epochs": 50,
                "image_size": [640, 640],
                "learning_rate": 0.01,
                "letter_box": True,
                "batch_size": 16,
            },
            "s": {
                "epochs": 50,
                "image_size": [640, 640],
                "learning_rate": 0.01,
                "letter_box": True,
                "batch_size": 16,
            },
            "m": {
                "epochs": 50,
                "image_size": [640, 640],
                "learning_rate": 0.01,
                "letter_box": True,
                "batch_size": 16,
            },
            "l": {
                "epochs": 50,
                "image_size": [640, 640],
                "learning_rate": 0.01,
                "letter_box": True,
                "batch_size": 16,
            },
            "x": {
                "epochs": 50,
                "image_size": [640, 640],
                "learning_rate": 0.01,
                "letter_box": True,
                "batch_size": 16,
            },
        }
    },
    "classification": {
        "yolov8": {
            "n": {
                "epochs": 50,
                "image_size": [224, 224],
                "learning_rate": 0.01,
                "letter_box": False,
                "batch_size": 16,
            },
            "s": {
                "epochs": 50,
                "image_size": [224, 224],
                "learning_rate": 0.01,
                "letter_box": False,
                "batch_size": 16,
            },
            "m": {
                "epochs": 50,
                "image_size": [224, 224],
                "learning_rate": 0.01,
                "letter_box": False,
                "batch_size": 16,
            },
            "l": {
                "epochs": 50,
                "image_size": [224, 224],
                "learning_rate": 0.01,
                "letter_box": False,
                "batch_size": 16,
            },
            "x": {
                "epochs": 50,
                "image_size": [224, 224],
                "learning_rate": 0.01,
                "letter_box": False,
                "batch_size": 16,
            },
        }
    },
    "instance_segmentation": {
        "yolov8": {
            "n": {
                "epochs": 50,
                "image_size": [640, 640],
                "learning_rate": 0.01,
                "letter_box": True,
                "batch_size": 16,
            },
            "s": {
                "epochs": 50,
                "image_size": [640, 640],
                "learning_rate": 0.01,
                "letter_box": True,
                "batch_size": 16,
            },
            "m": {
                "epochs": 50,
                "image_size": [640, 640],
                "learning_rate": 0.01,
                "letter_box": True,
                "batch_size": 16,
            },
            "l": {
                "epochs": 50,
                "image_size": [640, 640],
                "learning_rate": 0.01,
                "letter_box": True,
                "batch_size": 16,
            },
            "x": {
                "epochs": 50,
                "image_size": [640, 640],
                "learning_rate": 0.01,
                "letter_box": True,
                "batch_size": 16,
            },
        }
    },
}
