from waffle_hub import TaskType
from waffle_hub.schema.configs import TrainConfig

# Common
MODEL_TYPES = {
    TaskType.OBJECT_DETECTION: {"YOLOv5": list("sml")},
    TaskType.CLASSIFICATION: {"Classifier": list("sml")},
    TaskType.TEXT_RECOGNITION: {
        "TextRecognition": list("sml"),
        "LicencePlateRecognition": list("sml"),
    },
}

# Backend Specifics
DATA_TYPE_MAP = {
    TaskType.OBJECT_DETECTION: "COCODetectionDataset",
    TaskType.CLASSIFICATION: "COCOClassificationDataset",
    TaskType.TEXT_RECOGNITION: "COCOTextRecognitionDataset",
}

WEIGHT_PATH = {
    TaskType.OBJECT_DETECTION: {
        "YOLOv5": {
            "s": "temp/autocare_dlt/detectors/small/model.pth",
            "m": "temp/autocare_dlt/detectors/medium/model.pth",
            "l": "temp/autocare_dlt/detectors/large/model.pth",
        }
    },
    TaskType.CLASSIFICATION: {
        "Classifier": {
            "s": "temp/autocare_dlt/classifiers/small/model.pth",
            "m": "temp/autocare_dlt/classifiers/medium/model.pth",
            "l": "temp/autocare_dlt/classifiers/large/model.pth",
        }
    },
    TaskType.TEXT_RECOGNITION: {
        "TextRecognition": {
            "s": "temp/autocare_dlt/text_recognizers/small/model.pth",
            "m": "temp/autocare_dlt/text_recognizers/small/model.pth",
            "l": "temp/autocare_dlt/text_recognizers/small/model.pth",
        },
        "LicencePlateRecognition": {
            "s": "temp/autocare_dlt/text_recognizers/small/model.pth",
            "m": "temp/autocare_dlt/text_recognizers/small/model.pth",
            "l": "temp/autocare_dlt/text_recognizers/small/model.pth",
        },
    },
}

DEFAULT_PARAMS = {
    TaskType.OBJECT_DETECTION: {
        "YOLOv5": {
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
        }
    },
    TaskType.CLASSIFICATION: {
        "Classifier": {
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
        }
    },
    TaskType.TEXT_RECOGNITION: {
        "TextRecognition": {
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
        },
        "LicencePlateRecognition": {
            "s": TrainConfig(
                epochs=50,
                image_size=[240, 80],
                learning_rate=0.0002,
                letter_box=False,
                batch_size=256,
            ),
            "m": TrainConfig(
                epochs=50,
                image_size=[240, 80],
                learning_rate=0.0002,
                letter_box=False,
                batch_size=128,
            ),
            "l": TrainConfig(
                epochs=50,
                image_size=[240, 80],
                learning_rate=0.0002,
                letter_box=False,
                batch_size=64,
            ),
        },
    },
}
