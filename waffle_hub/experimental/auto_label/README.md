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
    bash download.sh
    ```

### Sample

#### Download Sample Image
```bash
mkdir sample_images
wget "https://static01.nyt.com/images/2023/01/27/multimedia/youpeople1-mwhp/youpeople1-mwhp-jumbo.jpg?quality=75&auto=webp" -O sample_images/youpeople1-mwhp-jumbo.jpg
```

#### Run Model

```bash
python -m waffle_hub.experimental.auto_label.grounding_dino \
    --draw \
    --config_file src/GroundingDINO_SwinT_OGC.py \
    --checkpoint_path src/groundingdino_swint_ogc.pth \
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