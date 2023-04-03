# Prepare Dataset
We provide [waffle_utils](https://github.com/snuailab/waffle_utils) to help you prepare your dataset. It will be automatically installed if you followed [Get Started](../getting_started.md).

## Download Sample Dataset
We've made sample dataset with mnist. You can download it with `waffle_utils`.

=== "Python"
    ```python
    from waffle_utils.file import io, network

    network.get_file_from_url(
        "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist.zip", 
        "mnist.zip"
    )
    io.unzip("mnist.zip", "mnist", create_directory=True)
    ```

=== "CLI"
    ```bash
    wu get_file_from_url \
        --url "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist.zip" \
        --file-path "mnist.zip"
    wu unzip --file-path "mnist.zip" --output-dir "mnist"
    ```

You'll see `mnist` in your current directory.
```
mnist
├── coco.json
└── images
    ├── 100.png
    ├── 10.png
    ├── 11.png
    ├── ...
```

## Create Dataset
Waffe Dataset is Object-oriented filesystem based dataset. Creating Dataset makes files and directories in your filesystem. You don't need to know the specific structure of dataset. `Dataset` will create it for you.

```
datasets/
└── mnist  # dataset name
    ├── annotations  # annotation meta files
    ├── categories  # category meta files
    ├── images  # image meta files
    ├── exports  # exported dataset
    ├── sets  # dataset split meta files
    └── raw  # raw image files
```

There are several ways to create Dataset.

### 1. Create empty Dataset

You can create empty Dataset with `Dataset.new` method.

=== "Python"
    ```python
    from waffle_utils.dataset import Dataset

    dataset = Dataset.new(
        name="mnist"
    )
    ```

| Argument | Type | Description |
| --- | --- | --- |
| name | str | Dataset name |
| root_dir | str | Dataset root directory. Default is `datasets` |

### 2. Import Existing Datasets
#### COCO Dataset

Waffle `Dataset` supports importing COCO dataset. You can import COCO dataset with `Dataset.from_coco` method.
Following example is importing [sample dataset](#create-dataset) we've downloaded.

=== "Python"
    ```python
    dataset = Dataset.from_coco(
        name="mnist",
        coco_file="mnist/coco.json",
        coco_root_dir="mnist/images",
    )
    ```

| Argument | Type | Description |
| --- | --- | --- |
| name | str | Dataset name |
| coco_file | str | COCO annotation file path |
| coco_root_dir | str | COCO image root directory |

| Return | Type | Description |
| --- | --- | --- |
| dataset | Dataset | Dataset object |

#### YOLO Dataset
Comming soon.

## Preprocess Dataset

After creating `Dataset`, you can preprocess it by `Dataset.split`, `Dataset.export`.

### 1. Split Dataset
Waffle Dataset supports split dataset. You can split dataset with `Dataset.split` method.

=== "Python"
    ```python
    dataset.split(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=123,
    )
    ```

| Argument | Type | Description |
| --- | --- | --- |
| train_ratio | float | Train set ratio |
| val_ratio | float | Validation set ratio |
| test_ratio | float | Test set ratio |
| seed | int | Random seed |

`Dataset.split` will create `train`, `val`, `test` set in `datasets/mnist/sets/` directory. Each set file contains image ids.

```
datasets/mnist/sets/
├── test.json
├── train.json
├── unlabeled.json
└── val.json
```

``` python
# train.json
[26, 40, 82, 87, ...]
```

### 2. Export Dataset
Waffle Dataset supports exporting dataset. You can export dataset with `Dataset.export` method.

=== "Python"
    ```python
    dataset_dir = dataset.export(
        export_format="YOLO_DETECTION"
    )
    # dataset_dir = "datasets/mnist/exports/YOLO_DETECTION"
    ```

| Argument | Type | Description |
| --- | --- | --- |
| export_format | Union[str, Dataset] | Export format. You can see supporting formats in `Dataset.Format`|

| Return | Type | Description |
| --- | --- | --- |
| dataset_dir | str | Exported dataset directory. It can be used as training argument. |

`Dataset.export` will create specified format dataset in `datasets/mnist/exports/[FORMAT]` directory.

```
datasets/mnist/exports/YOLO_DETECTION/
├── train
│   ├── images
│   └── labels
├── val
│   ├── images
│   └── labels
├── test
│   ├── images
│   └── labels
└── data.yaml
```