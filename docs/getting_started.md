## Install

### 🐍 Python
Waffle has been tested with Python `3.9`, `3.10`. <br>
Install any of the above versions you want to use.

### 🔥 PyTorch
Waffle has been tested with pytorch `1.13.1`. <br>
First, install pytorch and torchvision. See [pytorch.org](https://pytorch.org/get-started/previous-versions/) for details.

### 🧇 Waffle
You can install Waffle Hub with pip or from source.

=== "PyPI (recommended)"

    ``` bash
    pip install -U waffle_hub
    ```

=== "GitHub (advanced)"

    ``` bash
    git clone https://github.com/snuailab/waffle_hub.git
    cd waffle_hub
    pip install -e .
    ```

### 🛠️ Test your installation
``` bash
python -c "import waffle_hub; print(waffle_hub.__version__)"
```
