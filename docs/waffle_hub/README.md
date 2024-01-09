# Waffle Hub

`Waffle Hub` provide two key component classes: `Hub` and `Dataset`.

## [Hub](hub/hub.md)
`Hub` provides same interface for various powerfull Deep Learning Frameworks. Here is our brief system architecture.

<img src="../assets/hub_flow.png">

Each input and output adapter is responsible for converting our interface to the framework's interface. For example, [`Ultralytics`](https://github.com/ultralytics/ultralytics) uses `imgsz` for image size parameter, but [`detectron2`](https://github.com/facebookresearch/detectron2) uses `IMAGE_SIZE`. So, we need to convert our interface to the framework's interface. `waffle_hub` provides `InputAdapter` and `OutputAdapter` for this purpose.

## [Dataset](dataset/dataset.md)
`Dataset` class support many types of data format such as `coco`, `yolo`. You can use it to convert dataset or manage dataset.
