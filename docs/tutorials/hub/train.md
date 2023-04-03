# Train

After you [create `Hub`](./index.md), you can train it using `Hub.train` method. As creating `Hub`, we provide same interface for training. Following example is training with `UltralyticsHub` and [Sample Dataset](../prepare_dataset.md) we've created.

=== "Python"
    
    ``` python
    from waffle_hub.hub.adapter.ultralytics import UltralyticsHub

    hub = UltralyticsHub.load("digit_detector")
    train_callback = hub.train(
        dataset_path="datasets/mnist/exports/YOLO_DETECTION",
        epochs=10,
        batch_size=16,
        image_size=320,
        device="0",
        workers=2,
        seed=0,
        hold=True
    )
    ```

    | Argument | Type | Description |
    | --- | --- | --- |
    | dataset_path | str | Dataset path. Result of `Dataset.export`. |
    | epochs | int | Number of epochs |
    | batch_size | int | Batch size |
    | image_size | int | Image size |
    | device | str | Device to train |
    | workers | int | Number of workers |
    | seed | int | Random seed |
    | hold | bool | Hold training. This is an arguement for people using waffle as a SDK. If it is `False`, it will be excecuted by a thread.|

    | Return | Type | Description |
    | --- | --- | --- |
    | callback | [TrainCallback](../callbacks/#traincallback) | Callback object |
