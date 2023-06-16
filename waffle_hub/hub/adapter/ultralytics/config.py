from waffle_hub.schema.configs import TrainConfig

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

DEFAULT_PARAMS = {
    "object_detection": {
        "yolov8": {
            "n": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=64,
            ),
            "s": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=32,
            ),
            "m": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=16,
            ),
            "l": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=8,
            ),
            "x": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=8,
            ),
        }
    },
    "classification": {
        "yolov8": {
            "n": TrainConfig(
                epochs=50,
                image_size=[224, 224],
                learning_rate=0.01,
                letter_box=False,
                batch_size=512,
            ),
            "s": TrainConfig(
                epochs=50,
                image_size=[224, 224],
                learning_rate=0.01,
                letter_box=False,
                batch_size=256,
            ),
            "m": TrainConfig(
                epochs=50,
                image_size=[224, 224],
                learning_rate=0.01,
                letter_box=False,
                batch_size=128,
            ),
            "l": TrainConfig(
                epochs=50,
                image_size=[224, 224],
                learning_rate=0.01,
                letter_box=False,
                batch_size=64,
            ),
            "x": TrainConfig(
                epochs=50,
                image_size=[224, 224],
                learning_rate=0.01,
                letter_box=False,
                batch_size=64,
            ),
        }
    },
    "instance_segmentation": {
        "yolov8": {
            "n": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=32,
            ),
            "s": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=16,
            ),
            "m": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=8,
            ),
            "l": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=4,
            ),
            "x": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=4,
            ),
        }
    },
}
