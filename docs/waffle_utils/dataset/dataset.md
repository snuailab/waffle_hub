<!-- 
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
 -->

# Dataset
Waffe Dataset is Object-oriented filesystem based dataset. Creating Dataset makes files and directories in your filesystem. You don't need to know the specific structure of dataset. `Dataset` will create it for you.

## Properties

| Name | Type | Description |
| --- | --- | --- |
| name | str | Dataset name |
| root_dir | Path | Root Directory |
| dataset_dir | Path | Dataset Directory |
| raw_image_dir | Path | Raw Image Directory |
| image_dir | Path | Image Directory |
| annotation_dir | Path | Annotation Directory |
| prediction_dir | Path | Prediction Directory |
| category_dir | Path | Category Directory |
| export_dir | Path | Export Directory |
| set_dir | Path | Set Directory |
| train_set_file | Path | Train Set File |
| val_set_file | Path | Val Set File |
| test_set_file | Path | Test Set File |
| unlabeled_set_file | Path | Unlabeled Set File |
| classes | list[str] | Classes names |

## Methods

### `new`

Create new dataset.

| Argument | Type | Description |
| --- | --- | --- |
| name | str | Dataset name |
| root_dir | str | Dataset root directory. |

### `load`

Load Dataset

| Argument | Type | Description |
| --- | --- | --- |
| name | str | Dataset name that Waffle Created |
| root_dir | str | Dataset root directory. |

### `clone`

<!-- """Clone Existing Dataset

Args:
    src_name (str):
        Dataset name to clone.
        It should be Waffle Created Dataset.
    name (str): New Dataset name
    src_root_dir (str, optional): Source Dataset root directory. Defaults to None.
    root_dir (str, optional): New Dataset root directory. Defaults to None.

Raises:
    FileNotFoundError: if source dataset does not exist.
    FileExistsError: if new dataset name already exist.

Returns:
    Dataset: Dataset Class
""" -->

Clone Existing Dataset

| Argument | Type | Description |
| --- | --- | --- |
| src_name | str | Dataset name to clone. It should be Waffle Created Dataset. |
| name | str | New Dataset name |
| src_root_dir | str | Source Dataset root directory. |
| root_dir | str | New Dataset root directory. |

### `from_coco`
<!-- """Import Dataset from coco format.

Args:
    name (str): Dataset name.
    coco_file (str): Coco json file path.
    coco_root_dir (str): Coco image root directory.
    root_dir (str, optional): Dataset root directory. Defaults to None.

Raises:
    FileExistsError: if new dataset name already exist.

Returns:
    Dataset: Dataset Class
""" -->

Import Dataset from coco format.

| Argument | Type | Description |
| --- | --- | --- |
| name | str | Dataset name. |
| coco_file | str | Coco json file path. |
| coco_root_dir | str | Coco image root directory. |
| root_dir | str | Dataset root directory. |

### `get_images`
<!-- """Get "Image"s.

Args:
    image_ids (list[int], optional): id list. None for all "Image"s. Defaults to None.
    labeled (bool, optional): get labeled images. False for unlabeled images. Defaults to True.

Returns:
    list[Image]: "Image" list
""" -->

Get "[Image](../dataset/field.md#image)"s.

| Argument | Type | Description |
| --- | --- | --- |
| image_ids | list[int] | id list. None for all "Image"s. |
| labeled | bool | get labeled images. False for unlabeled images. |

| Return | Type | Description |
| --- | --- | --- |
| list[Image] | "Image" list |

### `get_categories`
<!-- """Get "Category"s.

Args:
    category_ids (list[int], optional): id list. None for all "Category"s. Defaults to None.

Returns:
    list[Category]: "Category" list
""" -->

Get "[Category](../dataset/field.md#category)"s.

| Argument | Type | Description |
| --- | --- | --- |
| category_ids | list[int] | id list. None for all "Category"s. |

| Return | Type | Description |
| --- | --- | --- |
| list[Category] | "Category" list |

### `get_annotations`
<!-- """Get "Annotation"s.

Args:
    image_id (int, optional): image id. None for all "Annotation"s. Defaults to None.

Returns:
    list[Annotation]: "Annotation" list
""" -->

Get "[Annotation](../dataset/field.md#annotation)"s.

| Argument | Type | Description |
| --- | --- | --- |
| image_id | int | image id. None for all "Annotation"s. |

| Return | Type | Description |
| --- | --- | --- |
| list[Annotation] | "Annotation" list |

### `split`
<!-- """Split Dataset to train, validation, test, (unlabeled) sets.

Args:
    train_ratio (float): train num ratio (0 ~ 1).
    val_ratio (float, optional): val num ratio (0 ~ 1).
    test_ratio (float, optional): test num ratio (0 ~ 1).
    seed (int, optional): random seed. Defaults to 0.
""" -->

Split Dataset to train, validation, test, (unlabeled) sets.

| Argument | Type | Description |
| --- | --- | --- |
| train_ratio | float | train num ratio (0 ~ 1). |
| val_ratio | float | val num ratio (0 ~ 1). |
| test_ratio | float | test num ratio (0 ~ 1). |
| seed | int | random seed. |

### `export`
<!-- """Export Dataset to Specific data formats

Args:
    export_format (Union[str, Format]): export format. one of {list(map(lambda x: x.name, Format))}.

Returns:
    str: exported dataset directory
""" -->

Export Dataset to Specific data formats

| Argument | Type | Description |
| --- | --- | --- |
| export_format | Union[str, Format] | export format. one of [YOLO_DETECTION, YOLO_CLASSIFICATION, COCO_DETECTION] |

| Return | Type | Description |
| --- | --- | --- |
| str | exported dataset directory |

