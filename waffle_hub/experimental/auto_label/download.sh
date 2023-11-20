git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e GroundingDINO

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P ./src
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4 -P ./src

python extract_frames.py --video_path ./src/people-detection.mp4 --output_dir ./src/frames --interval 200
python auto_label.py \
    --source ./src/frames \
    --text_prompt_file ./src/prompt.json \
    --class_names_file ./src/class_names.json \
    --waffle_dataset_name autolabel_sample \
    --draw \
    --device cpu
