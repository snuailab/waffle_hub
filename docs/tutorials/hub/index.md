# Hub
`waffle_hub` provides same interface for various powerfull Deep Learning Frameworks. Here is our brief system architecture.

<img src="https://raw.githubusercontent.com/snuailab/assets/main/waffle/docs/diagrams/hub_flow.png">

Each input and output adapter is responsible for converting our interface to the framework's interface. For example, [`Ultralytics`](https://github.com/ultralytics/ultralytics) uses `imgsz` for image size parameter, but [`detectron2`](https://github.com/facebookresearch/detectron2) uses `IMAGE_SIZE`. So, we need to convert our interface to the framework's interface. `waffle_hub` provides `InputAdapter` and `OutputAdapter` for this purpose.

## Usage
Waffle Hub is also based on Object-oriented filesystem like [Waffle Dataset](../prepare_dataset.md). You can create `Hub` with `Hub.new` method.

### Ultralytics Hub

