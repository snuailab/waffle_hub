![header](https://github.com/snuailab/assets/blob/main/snuailab/full/snuAiLab.black.300ppi.png?raw=true)

# Waffle Hub
- waffle 의 training core
- 다양한 backend framework 를 제공
- SNUAILAB 데이터 형식으로 모든 backend framework 를 사용 가능
<img src="https://github.com/snuailab/assets/blob/main/waffle/docs/diagrams/hub_flow.png?raw=true" width="100%" />
<!--
# Usage
### Query Format
```python
# "wh help" for available backends.
# "wh <backend> help" for available tasks.
# "wh <backend> <task> help" for available jobs.
# "wh <backend> <task> <job> help" for available parameters.
wh <backend> <task> <job> \  # base query
   name=<exp_name> \ 
   <param_key>=<value>  # params
```
### Basic Use
```python
wh yolo detection train \
   name=my_model \
   model=yolov8s \
   image_dir=./dataset/mnist/images \
   coco_file=./dataset/mnist/ann.json
   
wh mm pose_estimation train \
   name=my_model \
   model=DeepPose \
   image_dir=./dataset/mnist/images \
   coco_file=./dataset/mnist/ann.json
```
### Advanced Use
```python
# check "wh yolo train detection help" for available parameters.
wh yolo detection train \  
   name=my_model \
   model=yolov8s \
   image_dir=./dataset/mnist/images \
   coco_file=./dataset/mnist/ann.json \
   epoch=100 \
   patience=10 \
   batch=32 \
   learning_rate=0.001 \
   pretrained_weight=base_ckpt.pt \
   letter_box=True \
   r=128 \
   g=128 \
   b=128 \
   iou_threshold=0.6 \
   conf_threshold=0.8
```
or simply
```python
wh yolo detection train \
   name=my_model \
   config_from=config.yaml \
   <override>
```
or from pretrained model
```python
wh yolo detection train \
   name=my_model \
   start_from=model.pt \
   <override>
```
-->
