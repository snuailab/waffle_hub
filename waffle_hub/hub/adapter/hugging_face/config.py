from waffle_hub.schema.configs import TrainConfig

# Common
MODEL_TYPES = {
    "object_detection": {
        "DETA": {
            "base": "jozhang97/deta-resnet-50",
        },
        "DETR": {
            "base": "facebook/detr-resnet-50",
            "large": "facebook/detr-resnet-101",
        },
        "YOLOS": {
            "tiny": "hustvl/yolos-tiny",
        },
    },
    "classification": {
        "ViT": {
            "tiny": "WinKawaks/vit-tiny-patch16-224",
            "base": "google/vit-base-patch16-224",
        }
    },
}

DEFAULT_PARAMAS = {
    "object_detection": {
        "DETA": {
            "base": 
                TrainConfig(
                    epochs=50, 
                    image_size=[800, 800], 
                    learning_rate=5e-05, # TODO: implement letter_box
                    letter_box=True, 
                    batch_size=1
                )
        },
        "DETR": {
            "base": 
                TrainConfig(
                    epochs=50,
                    image_size=[800, 800],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=1,
                ),
            "large": 
                TrainConfig(
                    epochs=50,
                    image_size=[800, 800],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=1,
                ),
        },
        "YOLOS": {
            "tiny": 
                TrainConfig(
                    epochs=50,
                    image_size=[800, 800],
                    learning_rate=5e-05,
                    letter_box=True,  # TODO: implement letter_box
                    batch_size=16,
                ),
        },
    },
    "classification": {
        "ViT": {
            "tiny": 
                TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                ),
            "base":
                TrainConfig(
                    epochs=50,
                    image_size=[224, 224],
                    learning_rate=5e-05,
                    letter_box=False,
                    batch_size=128,
                )
        },
    },
}
