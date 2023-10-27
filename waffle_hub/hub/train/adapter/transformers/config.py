from waffle_dough.type.task_type import TaskType
from waffle_hub.schema.configs import TrainConfig
from waffle_hub.utils.utils import CaseInsensitiveDict

# Common
MODEL_TYPES = CaseInsensitiveDict(
    {
        TaskType.OBJECT_DETECTION: {
            "DETR": {
                "base": "facebook/detr-resnet-50",
                "large": "facebook/detr-resnet-101",
                "conditional": "microsoft/conditional-detr-resnet-50",
                "deformable": "SenseTime/deformable-detr",
            },
            "DETA": {
                "resnet": "jozhang97/deta-resnet-50",
                "swin": "jozhang97/deta-swin-large",
            },
            "YOLOS": {
                "base": "hustvl/yolos-base",
                "tiny": "hustvl/yolos-tiny",
                "small": "hustvl/yolos-small",
            },
        },
        TaskType.CLASSIFICATION: {
            "ResNet": {
                "50": "microsoft/resnet-50",
                "18": "microsoft/resnet-18",
                "101": "microsoft/resnet-101",
                "152": "microsoft/resnet-152",
            },
            "ViT": {
                "base": "google/vit-base-patch16-224",
                "tiny": "WinKawaks/vit-tiny-patch16-224",
                "large": "google/vit-large-patch16-224",
            },
            "ConvNextV2": {
                "base": "facebook/convnextv2-base-22k-224",
                "tiny": "facebook/convnextv2-tiny-22k-224",
                "large": "facebook/convnextv2-large-22k-224",
                "huge": "facebook/convnextv2-huge-22k-224",
            },
            "Swinv2": {
                "base": "microsoft/swinv2-base-patch4-window8-256",
                "tiny": "microsoft/swinv2-tiny-patch4-window8_256",
                "small": "microsoft/swinv2-small-patch4-window8_256",
                "large": "microsoft/swinv2-large-patch4-window12-192-22k",
            },
        },
    }
)

DEFAULT_PARAMS = CaseInsensitiveDict(
    {
        TaskType.OBJECT_DETECTION: {
            "DETR": {
                "base": TrainConfig(
                    epochs=50,
                    image_size=[800, 800],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=16,
                ),
                "large": TrainConfig(
                    epochs=50,
                    image_size=[800, 800],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=16,
                ),
                "conditional": TrainConfig(
                    epochs=50,
                    image_size=[800, 800],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=16,
                ),
                "deformable": TrainConfig(
                    epochs=50,
                    image_size=[800, 800],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=16,
                ),
            },
            "DETA": {
                "resnet": TrainConfig(
                    epochs=50,
                    image_size=[800, 800],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=16,
                ),
                "swin": TrainConfig(
                    epochs=50,
                    image_size=[800, 800],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=16,
                ),
            },
            "YOLOS": {
                "base": TrainConfig(
                    epochs=50,
                    image_size=[640, 640],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=16,
                ),
                "tiny": TrainConfig(
                    epochs=50,
                    image_size=[640, 640],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=16,
                ),
                "small": TrainConfig(
                    epochs=50,
                    image_size=[640, 640],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=16,
                ),
            },
        },
        TaskType.CLASSIFICATION: {
            "ResNet": {
                "50": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "18": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "101": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "152": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
            },
            "ViT": {
                "base": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "tiny": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "large": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
            },
            "ConvNextV2": {
                "base": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "tiny": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "large": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "huge": TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
            },
            "Swinv2": {
                "base": TrainConfig(
                    epochs=50,
                    image_size=[256, 256],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "tiny": TrainConfig(
                    epochs=50,
                    image_size=[256, 256],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "small": TrainConfig(
                    epochs=50,
                    image_size=[256, 256],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
                "large": TrainConfig(
                    epochs=50,
                    image_size=[256, 256],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
            },
        },
    }
)
