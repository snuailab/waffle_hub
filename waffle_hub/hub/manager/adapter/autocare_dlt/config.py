from waffle_hub.schema.configs import TrainConfig
from waffle_hub.type import TaskType

# Common
MODEL_TYPES = {
    TaskType.OBJECT_DETECTION: {
        "YOLOv5": {
            "s": "",
            "m": "",
            "l": "",
        },
    },
    TaskType.CLASSIFICATION: {
        "Classifier": {
            "s": "",
            "m": "",
            "l": "",
        },
    },
    TaskType.TEXT_RECOGNITION: {
        "TextRecognition": {
            "s": "",
            "m": "",
            "l": "",
        },
        "LicencePlateRecognition": {
            "s": "",
            "m": "",
            "l": "",
        },
    },
    TaskType.SEMANTIC_SEGMENTATION: {
        "Segmenter": {
            "m": "",
        },
    },
}

# Backend Specifics
DATA_TYPE_MAP = {
    TaskType.OBJECT_DETECTION: "COCODetectionDataset",
    TaskType.CLASSIFICATION: "COCOClassificationDataset",
    TaskType.TEXT_RECOGNITION: "COCOTextRecognitionDataset",
    TaskType.SEMANTIC_SEGMENTATION: "COCOSegmentationDataset",
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
            "m": "temp/autocare_dlt/text_recognizers/medium/model.pth",
            "l": "temp/autocare_dlt/text_recognizers/large/model.pth",
        },
        "LicencePlateRecognition": {
            "s": "temp/autocare_dlt/text_recognizers/small/model.pth",
            "m": "temp/autocare_dlt/text_recognizers/medium/model.pth",
            "l": "temp/autocare_dlt/text_recognizers/large/model.pth",
        },
    },
    TaskType.SEMANTIC_SEGMENTATION: {
        "Segmenter": {
            "m": "temp/autocare_dlt/semantic_segmentation/medium/model.pth",
        }
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
    TaskType.SEMANTIC_SEGMENTATION: {
        "Segmenter": {
            "m": TrainConfig(
                epochs=50,
                image_size=[640, 640],
                learning_rate=0.01,
                letter_box=True,
                batch_size=4,
            ),
        }
    },
}
