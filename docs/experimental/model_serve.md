# Model Serve

## Overview
We provide a simple way to serve your model. You can serve your model with `waffle_utils.model_serve.serve` method. It will create a simple web server with your model. You can access your model with HTTP request. We only provide one batch inference. We will provide more features in the future.

## Prerequisites

You need to install `FastAPI` and `uvicorn` to run server additionally. This is experimental feature, so we didn't add it to `requirements.txt`.

=== "CLI"

    ``` bash
    pip install fastapi uvicorn
    ```

## Run server

=== "CLI"

    ``` bash
    python -m waffle_hub.experimental.serve \
        --name test  # hub name \
        --root_dir hubs  # root directory \
        --host 0.0.0.0  # host name \
        --port 8000  # port number \
        --device 0  # cuda device id or cpu
    ```

## Client

=== "Python"

    ``` python
    import requests

    image_file_path = "test.jpg"
    url = "http://localhost:8000/predict"

    with open(image_file_path, "rb") as f:
        image = f.read()

    response = requests.post(url, files={"file": image})

    print(response.json())
    ```

=== "CURL"

    ``` bash
    curl -X POST \
        -F "file=@test.jpg" \
        http://localhost:8000/predict
    ```