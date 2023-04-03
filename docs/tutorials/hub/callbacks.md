# Callbacks

Each [`train`](train.md), [`inference`](inference.md), [`evaluation`](evaluate.md), [`export`](export.md) job will return `Callback` object. `Callback` object has several methods to get information about job. Basically, Every `Callback` class inherits `ThreadProgressCallback` class.

## ThreadProgressCallback

### Attributes
None

### Methods
#### `__init__`

Initialize `ThreadProgressCallback` object.

| Argument | Type | Description |
| --- | --- | --- |
| total_steps | int | Total steps |

#### `get_progress`

Get progress of job.

| Return | Type | Description |
| --- | --- | --- |
| progress | float | Progress (0 ~ 1)|

#### `is_finished`

Check if job is finished.

| Return | Type | Description |
| --- | --- | --- |
| is_finished | bool | If job is finished |

#### `get_remaining_time`

Get remaining time of job.

| Return | Type | Description |
| --- | --- | --- |
| remaining_time | float | Remaining time (second) |

#### `update`

Update progress of job.

| Argument | Type | Description |
| --- | --- | --- |
| step | int | Current step |

#### `register_thread`

Register thread to `ThreadProgressCallback` object.

| Argument | Type | Description |
| --- | --- | --- |
| thread | threading.Thread | Thread to register |

#### `start`

Start thread.

#### `join`

Join thread.

#### `force_finish`

Make job finished.

## TrainCallback

Return object of [`Hub.train`](../train/#train) method.

### Attributes

| Name | Type | Description |
| --- | --- | --- |
| best_ckpt_file | str | Best checkpoint file path |
| last_ckpt_file | str | Last checkpoint file path |
| result_dir | str | Result directory path |
| metric_file | str | Metric file path |

### Methods

#### `__init__`

Initialize `TrainCallback` object.

| Argument | Type | Description |
| --- | --- | --- |
| total_steps | int | Total steps |
| get_metric_func | Callable | Function to get metric |

#### `get_metrics`

Get metrics.

| Return | Type | Description |
| --- | --- | --- |
| metrics | list[list[dict]] | Metrics |

=== "Example"

    ``` python
    [
        [
            {'tag': 'loss', 'value': 0.123}, 
            {'tag': 'accuracy', 'value': 0.987},
            ...
        ],  # epoch 1
        [],  # epoch 2
        ...
    ]
    ```

## InferenceCallback

Return object of [`Hub.inference`](inference.md) method.

### Attributes

| Name | Type | Description |
| --- | --- | --- |
| inference_dir | str | Inference directory path |
| draw_dir | str | Draw directory path |

## EvaluationCallback
None

## ExportCallback

Return object of [`Hub.export`](export.md) method.

### Attributes

| Name | Type | Description |
| --- | --- | --- |
| export_file | str | Exported file path |
