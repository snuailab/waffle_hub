#!/bin/bash

# List of files to download
urls=(
    "https://raw.githubusercontent.com/snuailab/assets/main/waffle/fonts/gulim.ttc"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt"
)

echo "Downloading files..."

for url in "${urls[@]}"; do
    file=$(basename "$url")
    echo file
    if [ ! -f "$file" ]; then
        echo "Downloading $file..."
        wget "$url"
        echo "Downloaded $file"
    else
        echo "$file already exists. Skipping..."
    fi
done