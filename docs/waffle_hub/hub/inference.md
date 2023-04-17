# Inference

After you [train `Hub`](train.md), you can now inference it using `Hub.inference` method.

<!-- """Start Inference

Args:
    source (str): source directory
    recursive (bool, optional): recursive. Defaults to True.
    image_size (Union[int, list[int]], optional): image size. None for using training config. Defaults to None.
    letter_box (bool, optional): letter box. None for using training config. Defaults to None.
    batch_size (int, optional): batch size. Defaults to 4.
    confidence_threshold (float, optional): confidence threshold. Defaults to 0.25.
    iou_threshold (float, optional): iou threshold. Defaults to 0.5.
    half (bool, optional): half. Defaults to False.
    workers (int, optional): workers. Defaults to 2.
    device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
    draw (bool, optional): draw. Defaults to False.
    hold (bool, optional): hold. Defaults to True.


Raises:
    FileNotFoundError: if can not detect appropriate dataset.
    e: something gone wrong with ultralytics

Returns:
    InferenceCallback: inference callback
""" -->

=== "Python"
    
    ``` python
    from waffle_hub.hub.adapter.ultralytics import UltralyticsHub
    
    hub = UltralyticsHub.load("digit_detector")
    inference_callback = hub.inference(
        source="mnist",
        recursive=True,
        image_size=None,
        letter_box=None,
        batch_size=4,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        half=False,
        workers=2,
        device="0",
        draw=False,
        hold=True
    )
    ```

    | Argument | Type | Description | Default |
    | --- | --- | --- | --- |
    | source | str | source directory | |
    | recursive | bool | recursive. | True |
    | image_size | Union[int, list[int]] | image size. None for using training config. | None |
    | letter_box | bool | letter box. None for using training config. | None |
    | batch_size | int | batch size. | 4 |
    | confidence_threshold | float | confidence threshold. | 0.25 |
    | iou_threshold | float | iou threshold. | 0.5 |
    | half | bool | half. | False |
    | workers | int | workers. | 2 |
    | device | str | device. "cpu" or "gpu_id". | "0" |
    | draw | bool | draw. | False |
    | hold | bool | hold. | True |
    
    | Return | Type | Description |
    | --- | --- | --- |
    | inference_callback | InferenceCallback | Callback object |

## Inference Callback

## Properties

| Property | Type | Description |
| --- | --- | --- |
| inference_dir | str | Get the path of the result directory. |
| draw_dir | str | Get the path of the visualize directory. |

## Methods

| Method | Return Type | Description |
| --- | --- | --- |
| get_progress() | float | Get the progress of the task. (0 ~ 1) |
| is_finished() | bool | Check if the task has finished. |
| is_failed() | bool | Check if the task has failed. |
| get_remaining_time() | float | Get the remaining time of the task. (seconds) |
| update(step: int) | None | Update the progress of the task. (0 ~ total_steps) |
| force_finish() | None | Force the task to end. |
| register_thread(thread: threading.Thread) | None | Register the thread that is running the task. |
| start() | None | Start the thread that is running the task. |
| join() | None | Wait for the thread that is running the task to end. |
