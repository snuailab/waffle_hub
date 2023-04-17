import argparse
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T

import tqdm

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from waffle_utils.dataset import Dataset
from waffle_utils.file import io
from waffle_utils.log import initialize_logger

initialize_logger("logs/auto_label.log")

from waffle_hub.utils.data import ImageDataset
from waffle_hub.utils.draw import draw_results
from waffle_hub.schema.data import ImageInfo
from waffle_hub.schema.fields import Annotation


def parse_results(bboxes: torch.Tensor, confs: list[float], labels: list[str], image_info: ImageInfo, class2id: dict):

    W, H = image_info.input_shape
    left_pad, top_pad = image_info.pad
    ori_w, ori_h = image_info.ori_shape
    new_w, new_h = image_info.new_shape

    # cx cy w h (0~1) -> x1 y1 x2 y2 (0~1)
    cx, cy, w, h = torch.unbind(bboxes, dim=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

    parsed = []
    for (x1, y1, x2, y2), conf, label in zip(
        bboxes, confs, labels
    ):

        x1 = max(float((x1 * W - left_pad) / new_w * ori_w), 0)
        y1 = max(float((y1 * H - top_pad) / new_h * ori_h), 0)
        x2 = min(float((x2 * W - left_pad) / new_w * ori_w), ori_w)
        y2 = min(float((y2 * H - top_pad) / new_h * ori_h), ori_h)

        parsed.append(
            Annotation.object_detection(
                bbox=[x1, y1, x2 - x1, y2 - y1],
                area=float((x2 - x1) * (y2 - y1)),
                category_id=class2id[label],
                score=float(conf),
            )
        )
    return parsed


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image, captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    boxes = []
    labels = []
    confs = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        boxes.append(box.tolist())
        labels.append(pred_phrase)
        confs.append(logit.max().item())

    return torch.tensor(boxes), confs, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, default="autolabel_tmp/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default="autolabel_tmp/groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument("--source", "-s", type=str, required=True, help="path to image file or directory")
    parser.add_argument("--image_size", "-i", type=int, default=640, help="image size")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--draw", action="store_true", help="draw bounding boxes")

    parser.add_argument("--waffle_dataset_name", default=None, type=str, help="waffle dataset name")

    parser.add_argument(
        "--device", default="0", type=str, help="cuda device id or cpu"
    )
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model

    # text_prompt = args.text_prompt
    text_prompt = args.text_prompt
    classes = [text_prompt]
    class2id = {name: i for i, name in enumerate(classes)}
    id2class = {i: name for i, name in enumerate(classes)}
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold

    # device
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"

    # directory
    source_dir = args.source
    output_dir = Path(args.output_dir)
    draw_dir = output_dir / "draw"

    # load model
    model = load_model(config_file, checkpoint_path, device)

    # get dataloader
    dl = ImageDataset(image_dir=source_dir, image_size=args.image_size, letter_box=True).get_dataloader(1, 1)
    preprocess = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # coco format
    coco = {
        "categories": [{"id": i+1, "name": c, "supercategory": "object"} for i, c in id2class.items()],
        "images": [],
        "annotations": [],
    }
    image_id = 1
    annotation_id = 1

    for images, image_infos in tqdm.tqdm(dl, total=len(dl)):

        images = preprocess(images)
        
        # run model
        boxes, confs, labels = get_grounding_output(
            model, images, text_prompt, box_threshold, text_threshold, device
        )
        image_info = image_infos[0]  # TODO: batch
        if len(boxes) > 0:
            results = parse_results(boxes, confs, labels, image_info, class2id)
        else:
            results = []
        
        # save result as coco format
        file_name = Path(image_info.image_path).relative_to(source_dir)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": str(file_name),
                "width": image_info.ori_shape[0],
                "height": image_info.ori_shape[1],
            }
        )
        for result in results:
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": result.category_id+1,
                    "bbox": result.bbox,
                    "score": result.score,
                }
            )
            annotation_id += 1
        image_id += 1

        if args.draw:
            draw = draw_results(
                image_info.image_path,
                results,
                names=classes,
            )
            draw_path = draw_dir / file_name.with_suffix(".png")
            io.make_directory(draw_path.parent)
            cv2.imwrite(str(draw_path), draw)

    # save coco format
    io.save_json(coco, output_dir / "coco.json", create_directory=True)

    # make waffle dataset
    if args.waffle_dataset_name:
        Dataset.from_coco(
            name=args.waffle_dataset_name,
            coco_file=output_dir / "coco.json",
            coco_root_dir=source_dir,
            root_dir=output_dir,
        )
        print(f"Your waffle dataset has been saved to {output_dir}/{args.waffle_dataset_name}")
