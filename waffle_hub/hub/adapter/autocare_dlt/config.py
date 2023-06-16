from waffle_hub.schema.configs import TrainConfig

# Common
MODEL_TYPES = {
    "object_detection": {"YOLOv5": list("sml")},
    "classification": {"Classifier": list("sml")},
    "text_recognition": {"TextRecognition": list("sml"), "LicencePlateRecognition": list("sml")},
}

# Backend Specifics
DATA_TYPE_MAP = {
    "object_detection": "COCODetectionDataset",
    "classification": "COCOClassificationDataset",
    "text_recognition": "COCOTextRecognitionDataset",
}

WEIGHT_PATH = {
    "object_detection": {
        "YOLOv5": {
            "s": "temp/autocare_dlt/detectors/small/model.pth",
            "m": "temp/autocare_dlt/detectors/medium/model.pth",
            "l": "temp/autocare_dlt/detectors/large/model.pth",
        }
    },
    "classification": {
        "Classifier": {
            "s": "temp/autocare_dlt/classifiers/small/model.pth",
            "m": "temp/autocare_dlt/classifiers/medium/model.pth",
            "l": "temp/autocare_dlt/classifiers/large/model.pth",
        }
    },
    "text_recognition": {
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
    "object_detection": {
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
    "classification": {
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
    "text_recognition": {
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
