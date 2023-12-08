# Auto Labeling
This is a collection of auto labeling methods.

## 1. GroundingDINO

Auto labeling method based on [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).

### Prerequisites

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git

pip install -e GroundingDINO
pip install -U waffle_hub

mkdir autolabel_tmp
```

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
    --source sample_images/ \
    --text_prompt_file "src/prompt.json" \
    --class_names_file "src/class_names.json
```

#### Result

You can see the raw result in `outputs` and new WaffleDataset in your dataset root_dir .
