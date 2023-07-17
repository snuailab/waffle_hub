from waffle_hub import TaskType
from waffle_hub.schema.configs import TrainConfig

# Common
MODEL_TYPES = {
    TaskType.OBJECT_DETECTION: {
        "yolov8": {
            "n": "yolov8n.pt",
            "s": "yolov8s.pt",
            "m": "yolov8m.pt",
            "l": "yolov8l.pt",
            "x": "yolov8x.pt",
        },
        "yolov5": {
            "n": "yolov5nu.pt",
            "s": "yolov5su.pt",
            "m": "yolov5mu.pt",
            "l": "yolov5lu.pt",
            "x": "yolov5xu.pt",
            "n6": "yolov5n6u.pt",
            "s6": "yolov5s6u.pt",
            "m6": "yolov5m6u.pt",
            "l6": "yolov5l6u.pt",
            "x6": "yolov5x6u.pt",
        },
    },
    TaskType.CLASSIFICATION: {
        "yolov8": {
            "n": "yolov8n-cls.pt",
            "s": "yolov8s-cls.pt",
            "m": "yolov8m-cls.pt",
            "l": "yolov8l-cls.pt",
            "x": "yolov8x-cls.pt",
        }
    },
    TaskType.INSTANCE_SEGMENTATION: {
        "yolov8": {
            "n": "yolov8n-seg.pt",
            "s": "yolov8s-seg.pt",
            "m": "yolov8m-seg.pt",
            "l": "yolov8l-seg.pt",
            "x": "yolov8x-seg.pt",
        }
    },
    # "keypoint_detection": {"yolov8": list("nsmlx")},
}

# Backend Specifics
TASK_MAP = {
    TaskType.OBJECT_DETECTION: "detect",
    TaskType.CLASSIFICATION: "classify",
    TaskType.INSTANCE_SEGMENTATION: "segment"
    # "keypoint_detection": "pose"
}

DEFAULT_PARAMS = {
    TaskType.OBJECT_DETECTION: {
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
        },
        "yolov5": {
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
            "n6": TrainConfig(
                epochs=50,
                image_size=[1280, 1280],
                learning_rate=0.01,
                letter_box=True,
                batch_size=32,
            ),
            "s6": TrainConfig(
                epochs=50,
                image_size=[1280, 1280],
                learning_rate=0.01,
                letter_box=True,
                batch_size=16,
            ),
            "m6": TrainConfig(
                epochs=50,
                image_size=[1280, 1280],
                learning_rate=0.01,
                letter_box=True,
                batch_size=8,
            ),
            "l6": TrainConfig(
                epochs=50,
                image_size=[1280, 1280],
                learning_rate=0.01,
                letter_box=True,
                batch_size=4,
            ),
            "x6": TrainConfig(
                epochs=50,
                image_size=[1280, 1280],
                learning_rate=0.01,
                letter_box=True,
                batch_size=2,
            ),
        },
    },
    TaskType.CLASSIFICATION: {
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
    TaskType.INSTANCE_SEGMENTATION: {
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
