from pathlib import Path
from typing import List

import typer
from rich import print

from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.adapter.hugging_face import HuggingFaceHub
from waffle_hub.hub.adapter.tx_model import TxModelHub
from waffle_hub.hub.adapter.ultralytics import UltralyticsHub

dataset = typer.Typer(name="dataset")
hub = typer.Typer(name="hub")
app = typer.Typer()
app.add_typer(dataset)
app.add_typer(hub)

BACKEND_MAP = {
    "ultralytics": UltralyticsHub,
    "huggingface": HuggingFaceHub,
    "tx_model": TxModelHub,
}


@dataset.command(name="get_file_from_url")
def _get_file_from_url(url):
    print("get_file_from_url")


@hub.command(name="train")
def _train(backend):
    print("train")


@hub.command(name="new")
def _new(
    backend: str = typer.Option(..., help="Backend to use"),
    name: str = typer.Option(..., help="Name of the hub"),
    task: str = typer.Option(..., help="Task type"),
    model_type: str = typer.Option(..., help="Model type"),
    model_size: str = typer.Option(..., help="Model size"),
    categories: List[str] = typer.Option(..., help="Categories"),
    root_dir: str = typer.Option(..., help="Root directory"),
):
    if backend not in BACKEND_MAP:
        raise ValueError(f"Backend {backend} not found")

    BACKEND_MAP[backend].new(
        name=name,
        task=task,
        model_type=model_type,
        model_size=model_size,
        categories=categories,
        root_dir=root_dir,
    )


@hub.command(name="train")
def _train(
    backend: str = typer.Option(..., help="Backend to use"),
    name: str = typer.Option(..., help="Name of the hub"),
    root_dir: str = typer.Option(..., help="Root directory"),
    dataset_path: str = typer.Option(..., help="Dataset path"),
    epochs: int = typer.Option(..., help="Number of epochs"),
    batch_size: int = typer.Option(..., help="Batch size"),
    image_size: int = typer.Option(..., help="Image size"),
    learning_rate: float = typer.Option(None, help="Learning rate"),
    letter_box: bool = typer.Option(None, help="Letter box"),
    pretrained_model: str = typer.Option(None, help="Pretrained model"),
    device: str = typer.Option("0", help="Device"),
    workers: int = typer.Option(2, help="Number of workers"),
    seed: int = typer.Option(0, help="Seed"),
    verbose: bool = typer.Option(True, help="Verbose"),
    hold: bool = typer.Option(True, help="Hold"),
):
    if backend not in BACKEND_MAP:
        raise ValueError(f"Backend {backend} not found")

    hub = BACKEND_MAP[backend].load(
        name=name,
        root_dir=root_dir,
    )

    hub.train(
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=learning_rate,
        letter_box=letter_box,
        pretrained_model=pretrained_model,
        device=device,
        workers=workers,
        seed=seed,
        verbose=verbose,
        hold=hold,
    )


@hub.command(name="inference")
def _inference():
    pass


@hub.command(name="evaluate")
def _evalute():
    pass


@hub.command(name="export")
def _export():
    pass


if __name__ == "__main__":
    app()
