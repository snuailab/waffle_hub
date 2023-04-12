# Train

After you [create `Hub`](./hub.md), you can train it using `Hub.train` method. As creating `Hub`, we provide same interface for training. Following example is training with `UltralyticsHub` and [Sample Dataset](../../waffle_utils/dataset/dataset.md) we've created.


<!-- """Start Train

Args:
    dataset_path (str): dataset path
    epochs (int, optional): number of epochs. None to use default. Defaults to None.
    batch_size (int, optional): batch size. None to use default. Defaults to None.
    image_size (Union[int, list[int]], optional): image size. None to use default. Defaults to None.
    learning_rate (float, optional): learning rate. None to use default. Defaults to None.
    letter_box (bool, optional): letter box. None to use default. Defaults to None.
    pretrained_model (str, optional): pretrained model. None to use default. Defaults to None.
    device (str, optional): device. "cpu" or "gpu_id". Defaults to "0".
    workers (int, optional): number of workers. Defaults to 2.
    seed (int, optional): random seed. Defaults to 0.
    verbose (bool, optional): verbose. Defaults to True.
    hold (bool, optional): hold process. Defaults to True.

Raises:
    FileExistsError: if trained artifact exists.
    FileNotFoundError: if can not detect appropriate dataset.
    ValueError: if can not detect appropriate dataset.
    e: something gone wrong with ultralytics

Returns:
    TrainCallback: train callback
""" -->

=== "Python"
    
    ``` python
    from waffle_hub.hub.adapter.ultralytics import UltralyticsHub
    
    hub = UltralyticsHub.new("digit_detector")
    train_callback = hub.train(
        dataset_path="mnist",
        epochs=None,
        batch_size=None,
        image_size=None,
        learning_rate=None,
        letter_box=None,
        pretrained_model=None,
        device="0",
        workers=2,
        seed=0,
        verbose=True,
        hold=True
    )
    ```

    | Argument | Type | Description | Default |
    | --- | --- | --- | --- |
    | dataset_path | str | dataset path | |
    | epochs | int | number of epochs. None to use default. | None |
    | batch_size | int | batch size. None to use default. | None |
    | image_size | Union[int, list[int]] | image size. None to use default. | None |
    | learning_rate | float | learning rate. None to use default. | None |
    | letter_box | bool | letter box. None to use default. | None |
    | pretrained_model | str | pretrained model. None to use default. | None |
    | device | str | device. "cpu" or "gpu_id". | "0" |
    | workers | int | number of workers. | 2 |
    | seed | int | random seed. | 0 |
    | verbose | bool | verbose. | True |
    | hold | bool | hold process. | True |

    | Return | Type | Description |
    | --- | --- | --- |
    | callback | TrainCallback | Callback object |

## Train Callback

### Properties

| Property | Type | Description |
| --- | --- | --- |
| best_ckpt_file | str | Get the path of the best model. |
| last_ckpt_file | str | Get the path of the last model. |
| result_dir | str | Get the path of the result directory. |
| metric_file | str | Get the path of the metric file. |

### Methods

| Method | Return Type | Description |
| --- | --- | --- |
| get_metrics() | list[list[dict]] | Get the metrics of the task. (list of list of dict) |
| get_progress() | float | Get the progress of the task. (0 ~ 1) |
| is_finished() | bool | Check if the task has finished. |
| get_remaining_time() | float | Get the remaining time of the task. (seconds) |
| update(step: int) | None | Update the progress of the task. (0 ~ total_steps) |
| force_finish() | None | Force the task to end. |
| register_thread(thread: threading.Thread) | None | Register the thread that is running the task. |
| start() | None | Start the thread that is running the task. |
| join() | None | Wait for the thread that is running the task to end. |
