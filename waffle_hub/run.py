from typing import List

import typer
from rich import print

from waffle_hub.dataset.dataset import Dataset
from waffle_hub.hub.adapter.autocare_dlt import AutocareDLTHub
from waffle_hub.hub.adapter.hugging_face import HuggingFaceHub
from waffle_hub.hub.adapter.ultralytics import UltralyticsHub

dataset = typer.Typer(name="dataset")
hub = typer.Typer(name="hub")
app = typer.Typer()
app.add_typer(dataset)
app.add_typer(hub)

BACKEND_MAP = {
    "ultralytics": UltralyticsHub,
    "huggingface": HuggingFaceHub,
    "autocare_dlt": AutocareDLTHub,
}

EXPORT_MAP = {
    "ultralytics": "YOLO",
    "huggingface": "HUGGINGFACE",
    "autocare_dlt": "AutocareDLT",
}


@hub.command(name="new")
def _new_hub(
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
def _train_hub(
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
def _inference_hub(
    backend: str = typer.Option(..., help="Backend to use"),
    name: str = typer.Option(..., help="Name of the hub"),
    root_dir: str = typer.Option(..., help="Root directory"),
    source: str = typer.Option(..., help="Source"),
    recursive: bool = typer.Option(False, help="Recursive"),
    image_size: int = typer.Option(None, help="Image size"),
    letter_box: bool = typer.Option(None, help="Letter box"),
    batch_size: int = typer.Option(4, help="Batch size"),
    confidence_threshold: float = typer.Option(0.25, help="Confidence threshold"),
    iou_threshold: float = typer.Option(0.5, help="IOU threshold"),
    half: bool = typer.Option(False, help="Half"),
    device: str = typer.Option("0", help="Device"),
    workers: int = typer.Option(2, help="Number of workers"),
    draw: bool = typer.Option(True, help="Draw"),
    hold: bool = typer.Option(True, help="Hold"),
):
    if backend not in BACKEND_MAP:
        raise ValueError(f"Backend {backend} not found")

    hub = BACKEND_MAP[backend].load(
        name=name,
        root_dir=root_dir,
    )

    hub.inference(
        source=source,
        recursive=recursive,
        image_size=image_size,
        letter_box=letter_box,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        half=half,
        device=device,
        workers=workers,
        draw=draw,
        hold=hold,
    )


@hub.command(name="evaluate")
def _evalute_hub(
    backend: str = typer.Option(..., help="Backend to use"),
    name: str = typer.Option(..., help="Name of the hub"),
    root_dir: str = typer.Option(..., help="Root directory"),
    dataset_root_dir: str = typer.Option(..., help="Dataset root directory"),
    dataset_name: str = typer.Option(..., help="Dataset name"),
    set_name: str = typer.Option("test", help="Set name"),
    batch_size: int = typer.Option(4, help="Batch size"),
    image_size: int = typer.Option(None, help="Image size"),
    letter_box: bool = typer.Option(None, help="Letter box"),
    confidence_threshold: float = typer.Option(0.25, help="Confidence threshold"),
    iou_threshold: float = typer.Option(0.5, help="IOU threshold"),
    half: bool = typer.Option(False, help="Half"),
    device: str = typer.Option("0", help="Device"),
    workers: int = typer.Option(2, help="Number of workers"),
    draw: bool = typer.Option(True, help="Draw"),
    hold: bool = typer.Option(True, help="Hold"),
):
    if backend not in BACKEND_MAP:
        raise ValueError(f"Backend {backend} not found")

    hub = BACKEND_MAP[backend].load(
        name=name,
        root_dir=root_dir,
    )

    hub.evaluate(
        dataset_name=dataset_name,
        dataset_root_dir=dataset_root_dir,
        set_name=set_name,
        batch_size=batch_size,
        image_size=image_size,
        letter_box=letter_box,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        half=half,
        device=device,
        workers=workers,
        draw=draw,
        hold=hold,
    )


@hub.command(name="export")
def _export_hub(
    backend: str = typer.Option(..., help="Backend to use"),
    name: str = typer.Option(..., help="Name of the hub"),
    root_dir: str = typer.Option(..., help="Root directory"),
    image_size: int = typer.Option(None, help="Image size"),
    batch_size: int = typer.Option(16, help="Batch size"),
    opset_version: int = typer.Option(11, help="Opset version"),
    half: bool = typer.Option(False, help="Half"),
    device: str = typer.Option("0", help="Device"),
    hold: bool = typer.Option(True, help="Hold"),
):
    if backend not in BACKEND_MAP:
        raise ValueError(f"Backend {backend} not found")

    hub = BACKEND_MAP[backend].load(
        name=name,
        root_dir=root_dir,
    )

    hub.export(
        image_size=image_size,
        batch_size=batch_size,
        opset_version=opset_version,
        half=half,
        device=device,
        hold=hold,
    )


@dataset.command(name="new")
def _new_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
    task: str = typer.Option(..., help="Task"),
):
    ds = Dataset.new(
        name=name,
        root_dir=root_dir,
        task=task,
    )


@dataset.command(name="split")
def _split_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
    train_ratio: float = typer.Option(0, help="Train ratio"),
    val_ratio: float = typer.Option(0, help="Validation ratio"),
    test_ratio: float = typer.Option(0, help="Test ratio"),
    method: str = typer.Option("random", help="Split Method"),
    seed: int = typer.Option(0, help="Seed"),
):
    ds = Dataset.load(
        name=name,
        root_dir=root_dir,
    )

    ds.split(train_ratio, val_ratio, test_ratio, method, seed)


@dataset.command(name="from_coco")
def _from_coco_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
    task: str = typer.Option(..., help="Task"),
    coco_file: str = typer.Option(..., help="COCO file"),
    coco_root_dir: str = typer.Option(..., help="COCO root directory"),
):
    Dataset.from_coco(
        name=name,
        root_dir=root_dir,
        coco_file=coco_file,
        coco_root_dir=coco_root_dir,
        task=task,
    )


@dataset.command(name="from_yolo")
def _from_yolo_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
    task: str = typer.Option(..., help="Task"),
    yaml_path: str = typer.Option(..., help="YOLO YAML path"),
):
    Dataset.from_yolo(
        name=name,
        root_dir=root_dir,
        yaml_path=yaml_path,
        task=task,
    )


@dataset.command(name="from_huggingface")
def _from_huggingface_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
    task: str = typer.Option(..., help="Task"),
    dataset_dir: str = typer.Option(..., help="Dataset directory"),
):
    Dataset.from_huggingface(
        name=name,
        root_dir=root_dir,
        task=task,
        dataset_dir=dataset_dir,
    )


@dataset.command(name="clone")
def _clone_dataset(
    src_name: str = typer.Option(..., help="Source name of the dataset"),
    src_root_dir: str = typer.Option(..., help="Source root directory"),
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
):
    Dataset.clone(
        src_name=src_name,
        src_root_dir=src_root_dir,
        name=name,
        root_dir=root_dir,
    )


@dataset.command(name="export")
def _export_dataset(
    data_type: str = typer.Option(..., help="Data type"),
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
):
    ds = Dataset.load(
        name=name,
        root_dir=root_dir,
    )

    ds.export(data_type)


@dataset.command(name="delete")
def _delete_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
):
    ds = Dataset.load(
        name=name,
        root_dir=root_dir,
    )

    ds.delete()


@dataset.command(name="get_split_ids")
def _get_split_ids(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
):
    ds = Dataset.load(
        name=name,
        root_dir=root_dir,
    )

    for set_name, set_ids in zip(["train", "val", "test", "unlabeled"], ds.get_split_ids()):
        print(f"{set_name}: {set_ids}")


@dataset.command(name="merge")
def _merge_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
    src_names: list[str] = typer.Option(..., help="Source name of the dataset"),
    src_root_dirs: list[str] = typer.Option(..., help="Source root directory"),
    task: str = typer.Option(..., help="Task"),
):
    Dataset.merge(
        name=name,
        root_dir=root_dir,
        src_names=src_names,
        src_root_dirs=src_root_dirs,
        task=task,
    )


@dataset.command(name="sample")
def _sample_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
    task: str = typer.Option(..., help="Task"),
):
    Dataset.sample(
        name=name,
        root_dir=root_dir,
        task=task,
    )


@dataset.command(name="get_images")
def _get_images_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
):
    ds = Dataset.load(
        name=name,
        root_dir=root_dir,
    )

    images = ds.get_images()

    for image in images:
        print(image.to_dict())


@dataset.command(name="get_annotations")
def _get_annotations_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
):
    ds = Dataset.load(
        name=name,
        root_dir=root_dir,
    )

    annotations = ds.get_annotations()

    for annotation in annotations:
        print(annotation.to_dict())


@dataset.command(name="get_categories")
def _get_categories_dataset(
    name: str = typer.Option(..., help="Name of the dataset"),
    root_dir: str = typer.Option(..., help="Root directory"),
):
    ds = Dataset.load(
        name=name,
        root_dir=root_dir,
    )

    categories = ds.get_categories()

    for category in categories:
        print(category.to_dict())


if __name__ == "__main__":
    app()
