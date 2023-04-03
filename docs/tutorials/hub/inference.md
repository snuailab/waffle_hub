# Inference

After you [train `Hub`](train.md), you can now inference it using `Hub.inference` method.

=== "Python"
    
    ``` python
    from waffle_hub.hub.adapter.ultralytics import UltralyticsHub

    hub = UltralyticsHub.load("digit_detector")
    inference_callback = hub.inference(
        source="mnist/images",
        recursive=True,
        image_size=None,
        letter_box=None,
        batch_size=16,
        confidence_threshold=0.25,
        iou_threshold=0.45,
        half=False,
        workers=2,
        device="0",
        draw=True,
        hold=True
    )
    ```

    | Argument | Type | Description |
    | --- | --- | --- |
    | source | str | dataset source. image file or image directory. |
    | recursive | bool | get images from directory recursively. Defaults to True. |
    | image_size | int | inference image size. None for same with train_config (recommended). |
    | letter_box | bool | letter box inference image. None for same with train_config (recommended). |
    | batch_size | int | batch size. |
    | confidence_threshold | float | confidence threshold. |
    | iou_threshold | float | iou threshold.(for object_detection) |
    | half | bool | use half precision. |
    | workers | int | number of workers for dataloader. |
    | device | str | device to inference. |
    | draw | bool | draw inference result. |
    | hold | bool | hold inference. This is an arguement for people using waffle as a SDK. If it is `False`, it will be excecuted by a thread.|

    | Return | Type | Description |
    | --- | --- | --- |
    | callback | [InferenceCallback](../callbacks/#inferencecallback) | Callback object |