
# Hub
Waffle Hub is also based on Object-oriented filesystem like [Waffle Dataset](../../waffle_utils/dataset/dataset.md).

## Attributes

| Name | Type | Description |
| --- | --- | --- |
| name | str | Hub name |
| root_dir | Path | Root Directory |
| task | str | Task name. |
| model_type | str | Model Type. |
| model_size | str | Model Size. |
| backend | str | Backend name. |
| version | str | Version |
| categories | list[dict] | Categories |
| hub_dir | Path | Hub(Model) Directory |
| artifact_dir | Path | Artifact Directory. This is raw output of each backend. |
| inference_dir | Path | Inference Results Directory |
| evaluation_dir | Path | Evaluation Results Directory |
| export_dir | Path | Export Results Directory |
| draw_dir | Path | Draw Results Directory |
| model_config_file | Path | Model Config yaml File |
| train_config_file | Path | Train Config yaml File |
| best_ckpt_file | Path | Best Checkpoint File |
| onnx_file | Path | Best Checkpoint File |
| last_ckpt_file | Path | Last Checkpoint File |
| metric_file | Path | Metric Csv File |

## Support Specifications

You can find which `task`, `model_type`, `model_size` are available in `[BackendName]Hub.MODEL_TYPES`.

=== "Example"
    ``` python
    # UltralyticsHub.MODEL_TYPES
    {
        task: {
            model_type: [model_size, ...]
        }
    }

    {
        "object_detection": {"yolov8": list("nsmlx")},
        "classification": {"yolov8": list("nsmlx")},
    }
    ```

## Methods

### `new`

Create new hub.

| Argument | Type | Description |
| --- | --- | --- |
| name | str | Hub name |
| task | str | Task name. |
| model_type | str | Model Type. |
| model_size | str | Model Size. |
| categories | list[dict] | Categories |
| root_dir | str | Root directory of hub repository. |

### `load`

Load existing hub.

| Argument | Type | Description |
| --- | --- | --- |
| name | str | Hub name |
| root_dir | str | Root directory of hub repository. |

### `from_model_config`

Create new hub from model config.

| Argument | Type | Description |
| --- | --- | --- |
| name | str | Hub name |
| model_config_file | str | Model config file path |
| root_dir | str | Root directory of hub repository. |

