# TODO: add cli tool
# import typer
# from rich import print

# from waffle_hub.hub import UltralyticsHub

# from waffle_utils.dataset import Dataset

# app = typer.Typer()

# BACKEND_MAP = {
#     "ultralytics": UltralyticsHub,
# }

# @app.command(name="create", )
# def _create(
#     backend: str = typer.Option(..., help='One of [' + ', '.join(BACKEND_MAP.keys()) + ']'),
#     name: str = type.Option(..., help="model name"),
#     task: str = type.Option(..., help="task"),
#     model_type: str = type.Option(..., help="model type"),
#     model_size: str = type.Option(..., help="model size"),
#     pretrained_model: str = type.Option(None, help="pretrained model"),
#     model_root_dir: str = type.Option(None, help="pretrained model"),
#     dataset_
#     epochs: int,
#     batch_size: int,
#     image_size: int,
#     device: str = "0",
#     workers: int = 2,
#     seed: int = 0,
#     verbose: bool = True,
# ):
#     if backend not in BACKEND_MAP:
#         raise ValueError(f"""
#         Backend {backend} is not supported.
#         Choose one of {list(BACKEND_MAP.keys())}
#         """)

#     # hub = BACKEND_MAP[backend](
#     #     name=name,
#     #     task=task
#     # )


# if __name__ == "__main__":
#     app()
