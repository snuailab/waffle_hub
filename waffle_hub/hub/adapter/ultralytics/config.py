from waffle_hub import TaskType
from waffle_hub.schema.configs import TrainConfig

# Common
MODEL_TYPES = {
    TaskType.OBJECT_DETECTION: {
        "yolov8": list("nsmlx"),
        "yolov5u": list("nsmlx"),
        "yolov5u6": list("nsmlx"),
    },
    TaskType.CLASSIFICATION: {"yolov8": list("nsmlx")},
    TaskType.INSTANCE_SEGMENTATION: {"yolov8": list("nsmlx")},
    # "keypoint_detection": {"yolov8": list("nsmlx")},
}

# Backend Specifics
TASK_MAP = {
    TaskType.OBJECT_DETECTION: "detect",
    TaskType.CLASSIFICATION: "classify",
    TaskType.INSTANCE_SEGMENTATION: "segment"
    # "keypoint_detection": "pose"
}

PRETRAINED_WEIGHT = {
    TaskType.OBJECT_DETECTION: {
        "yolov8": {
            "n" : "yolov8n.pt",
            "s" : "yolov8s.pt",
            "m" : "yolov8m.pt",
            "l" : "yolov8l.pt",
            "x" : "yolov8x.pt"
        },
        "yolov5u": {
            "n" : "yolov5nu.pt",
            "s" : "yolov5su.pt",
            "m" : "yolov5mu.pt",
            "l" : "yolov5lu.pt",
            "x" : "yolov5xu.pt"
        },
        "yolov5u6": {
            "n" : "yolov5n6u.pt",
            "s" : "yolov5s6u.pt",
            "m" : "yolov5m6u.pt",
            "l" : "yolov5l6u.pt",
            "x" : "yolov5x6u.pt"
        },
        "yolov6": {
            "n" : "yolov6n.pt",
            "s" : "yolov6s.pt",
            "m" : "yolov6m.pt",
            "l" : "yolov6l.pt",
            "x" : "yolov6x.pt"
        }
    },
    TaskType.CLASSIFICATION: {
        "yolov8": {
            "n" : "yolov8n-cls.pt",
            "s" : "yolov8s-cls.pt",
            "m" : "yolov8m-cls.pt",
            "l" : "yolov8l-cls.pt",
            "x" : "yolov8x-cls.pt"
        }
    },
    TaskType.INSTANCE_SEGMENTATION: {
        "yolov8": {
            "n" : "yolov8n-seg.pt",
            "s" : "yolov8s-seg.pt",
            "m" : "yolov8m-seg.pt",
            "l" : "yolov8l-seg.pt",
            "x" : "yolov8x-seg.pt"
        }
    },
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
        "yolov5u": {
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
        "yolov5u6": {
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
