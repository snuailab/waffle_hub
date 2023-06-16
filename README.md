<div align="center">
  <p>
    <a href="http://snuailab.ai/">
        <img width="75%" src="https://raw.githubusercontent.com/snuailab/assets/main/waffle/icons/waffle_banner.png">
    </a>
  </p>
</div>

Waffle is a framework that lets you use lots of different deep learning tools through just one interface. When it comes to MLOps (machine learning operations), you need to be able to keep up with all the new ideas in deep learning as quickly as possible. But it's hard to do that if you have to write all the code yourself. That's why we started a project to bring together different tools into one framework.

Experience the power of multiple deep learning frameworks at your fingertips with Waffle's seamless integration, unlocking limitless possibilities for your machine learning projects.

# Prerequisites
We've tested Waffle on the following environments:
| OS | Python | PyTorch | Device | Backend | Pass |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Ubuntu 20.04 | 3.9, 3.10 | 1.13.1 | CPU, GPU | All | [![Waffle Hub cpu test](https://github.com/snuailab/waffle_hub/actions/workflows/ci.yaml/badge.svg)](https://github.com/snuailab/waffle_hub/actions/workflows/ci.yaml) |
| Windows | 3.9, 3.10 | 1.13.1 | CPU, GPU | All | [![Waffle Hub cpu test](https://github.com/snuailab/waffle_hub/actions/workflows/ci.yaml/badge.svg)](https://github.com/snuailab/waffle_hub/actions/workflows/ci.yaml) |
| Ubuntu 20.04 | 3.9 | 1.13.1 | Multi GPU | Ultralytics |[![Waffle Hub multi-gpu(ddp) test on self-hosted runner](https://github.com/snuailab/waffle_hub/actions/workflows/ddp.yaml/badge.svg)](https://github.com/snuailab/waffle_hub/actions/workflows/ddp.yaml) |


We recommend using above environments for the best experience.

# Installation
1. Install pytorch and torchvision
    - [PyTorch and TorchVision](https://pytorch.org/get-started/locally/) (We recommend using 1.13.1)
2. Install Waffle Hub
    - `pip install -U waffle-hub`

# Example Usage
We provide both python module and CLI for Waffle Hub.

Following examples do the exact same thing.

## Python Module
```python
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
```

## CLI
```bash
wd sample --name mnist_classification --task classification
wd split --name mnist_classification --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
wd export --name mnist_classification --data-type YOLO

wh new --name my_classifier --task classification --model-type yolov8 --model-size n --categories [1,2]
wh train --name my_classifier --dataset-path datasets/mnist_classification/exports/YOLO --epochs 30 --batch-size 64 --image-size 64 --device cpu
wh inference --name my_classifier --source datasets/mnist_classification/exports/YOLO --draw --device cpu
```

See our [documentation](https://snuailab.github.io/waffle/) for more information!
