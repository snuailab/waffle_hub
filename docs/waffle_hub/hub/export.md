# Export

After you [train `Hub`](train.md), you can now export it using `Hub.export` method.

<!-- """Export Model

Args:
    image_size (Union[int, list[int]], optional): inference image size. None for same with train_config (recommended).
    batch_size (int, optional): dynamic batch size. Defaults to 16.
    opset_version (int, optional): onnx opset version. Defaults to 11.
    hold (bool, optional): hold or not.
        If True then it holds until task finished.
        If False then return Inferece Callback and run in background. Defaults to True.

Returns:
    ExportCallback: export callback
""" -->

=== "Python"

    ``` python
    from waffle_hub.hub.adapter.ultralytics import UltralyticsHub
    
    hub = UltralyticsHub.load("digit_detector")
    export_callback = hub.export(
        image_size=None,
        batch_size=16,
        opset_version=11,
        hold=True
    )
    ```

    | Argument | Type | Description | Default |
    | --- | --- | --- | --- |
    | image_size | Union[int, list[int]] | inference image size. None for same with train_config (recommended). | None |
    | batch_size | int | dynamic batch size. | 16 |
    | opset_version | int | onnx opset version. | 11 |
    | hold | bool | hold or not. If True then it holds until task finished. If False then return Inferece Callback and run in background. | True |

    | Return | Type | Description |
    | --- | --- | --- |
    | export_callback | ExportCallback | export callback |

## ExportCallback

## Properties

| Property | Type | Description |
| --- | --- | --- |
| export_file | str | Get the path of the result file. |

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
