# Auto Labeling
This is a collection of auto labeling methods.

## 1. GroundingDINO

Auto labeling method based on [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).

> Note💡 The performance of this method is not guaranteed.

> Warning⚠️ This implementation only support one word text prompt.

### Prerequisites

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git

pip install -e GroundingDINO
pip install -U waffle_hub

mkdir autolabel_tmp
```

=== "Download Small Model"

    ```bash
    wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth -O autolabel_tmp/groundingdino_swint_ogc.pth
    wget https://huggingface.co/ShilongLiu/GroundingDINO/raw/main/GroundingDINO_SwinT_OGC.cfg.py -O autolabel_tmp/GroundingDINO_SwinT_OGC.py
    ```

=== "Download Large Model"

    ```bash
    wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth -O autolabel_tmp/groundingdino_swinb_cogcoor.pth
    wget https://huggingface.co/ShilongLiu/GroundingDINO/raw/main/GroundingDINO_SwinB_OGC.cfg.py -O autolabel_tmp/GroundingDINO_SwinB.cfg.py
    ```

### Sample

#### Download Sample Image
```bash
mkdir sample_images
wget "https://static01.nyt.com/images/2023/01/27/multimedia/youpeople1-mwhp/youpeople1-mwhp-jumbo.jpg?quality=75&auto=webp" -O sample_images/youpeople1-mwhp-jumbo.jpg
```

#### Run Model

=== "Small Model"

    ```bash
    python -m waffle_hub.experimental.auto_label.grounding_dino \
        --draw \
        --config_file autolabel_tmp/GroundingDINO_SwinT_OGC.py \
        --checkpoint_path autolabel_tmp/groundingdino_swint_ogc.pth \
        --source sample_images/ \
        --output_dir outputs/ \
        --text_prompt person
    ```

=== "Large Model"

    ```bash
    python -m waffle_hub.experimental.auto_label.grounding_dino \
        --draw \
        --config_file autolabel_tmp/GroundingDINO_SwinB.cfg.py \
        --checkpoint_path autolabel_tmp/groundingdino_swinb_cogcoor.pth \
        --source sample_images/ \
        --output_dir outputs/ \
        --text_prompt person
    ```

#### Result

You can see the result in `outputs`.

```bash
outputs/
├── coco.json
└── draw
    └── youpeople1-mwhp-jumbo.png
```