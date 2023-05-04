from waffle_hub.schema.configs import TrainConfig

# Common
MODEL_TYPES = {
    "object_detection": {"YOLOv5": list("sml")},
    "classification": {"Classifier": list("sml")},
}

# Backend Specifics
DATA_TYPE_MAP = {
    "object_detection": "COCODetectionDataset",
    "classification": "COCOClassificationDataset",
}

WEIGHT_PATH = {
    "object_detection": {
        "YOLOv5": {
            "s": "temp/autocare_tx_model/detectors/small/model.pth",
            "m": "temp/autocare_tx_model/detectors/medium/model.pth",
            "l": "temp/autocare_tx_model/detectors/large/model.pth",
        }
    },
    "classification": {
        "Classifier": {
            "s": "temp/autocare_tx_model/classifiers/small/model.pth",
            "m": "temp/autocare_tx_model/classifiers/medium/model.pth",
            "l": "temp/autocare_tx_model/classifiers/large/model.pth",
        }
    },
}

DEFAULT_PARAMAS = {
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
}