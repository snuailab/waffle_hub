from typing import List

import typer
from rich import print

from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.adapter.ultralytics import UltralyticsHub

dataset = typer.Typer(name="dataset")
hub = typer.Typer(name="hub")
app = typer.Typer()
app.add_typer(dataset)
app.add_typer(hub)

BACKEND_MAP = {
    "ultralytics": UltralyticsHub,
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
    print(backend, name, task, model_type, model_size, categories, root_dir)
    print(categories)
    BACKEND_MAP[backend].new(
        name=name,
        task=task,
        model_type=model_type,
        model_size=model_size,
        categories=categories,
        root_dir=root_dir,
    )


if __name__ == "__main__":
    app()
