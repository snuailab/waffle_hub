from waffle_hub.dataset import Dataset

ds = Dataset.from_yolo("")
dataset = Dataset.sample(
    name="mnist_classification",
    task="classification",
)
dataset.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
export_dir = dataset.export("YOLO")
