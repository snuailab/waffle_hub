"""
How to use

python -m waffle_hub.experimental.serve \
    --name test  # hub name \
    --root_dir hubs  # root directory \
    --host 0.0.0.0  # host name \
    --port 8000  # port number \
    --device 0  # cuda device id or cpu
"""
import logging

import fastapi
import uvicorn

import cv2
import torch
import numpy as np

from waffle_utils.file import io 
from waffle_utils.log import initialize_logger
initialize_logger("logs/serve.log")

from waffle_hub.hub.adapter.ultralytics import UltralyticsHub
from waffle_hub.hub.model.wrapper import get_parser

app = fastapi.FastAPI()

# recive image file with post method
@app.post("/predict")
async def predict(file: fastapi.UploadFile = fastapi.File(...)):
    image = await file.read()
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    logging.info(f"start inference")

    try:
        result = model(image.to(device))
    except Exception as e:
        logging.error(f"Error: {e}")
        raise fastapi.HTTPException(status_code=500, detail=f"Error: {e}")
    
    logging.info(f"end inference")

    return result

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="hub name")
    parser.add_argument("--root_dir", type=str, default=None, help="root directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--device", default="0", type=str, help="cuda device id or cpu")
    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logging.info(f"using device: {device}")

    hub: UltralyticsHub = UltralyticsHub.load(args.name, root_dir=args.root_dir)
    logging.info(f"loaded hub: {hub.name}")

    train_config = io.load_yaml(hub.train_config_file)
    parser = get_parser(hub.task)(**train_config)

    model = hub.get_model(image_size=train_config.get("image_size", None), parser=parser)
    model.eval()
    model.to(device)

    uvicorn.run(app, host=args.host, port=args.port)
