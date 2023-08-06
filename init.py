from waffle_hub.dataset import Dataset
dataset = Dataset.sample(
  name = "mnist_classification",
  task = "classification",
)
dataset.split(
  train_ratio = 0.8,
  val_ratio = 0.1,
  test_ratio = 0.1
)
export_dir = dataset.export("YOLO")

from waffle_hub.hub import Hub
hub = Hub.new(
  name = "my_classifier",
  task = "classification",
  model_type = "yolov8",
  model_size = "n",
  categories = dataset.get_category_names(),
)
hub.train(
  dataset_path = export_dir,
  epochs = 30,
  batch_size = 64,
  image_size=64,
  device="cpu"
)
hub.inference(
  source=export_dir,
  draw=True,
  device="cpu"
)