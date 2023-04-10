# Waffle Dataset
We provide [waffle_utils](https://github.com/snuailab/waffle_utils) to help you prepare your dataset. It will be automatically installed if you followed [Get Started](../../getting_started.md).

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
