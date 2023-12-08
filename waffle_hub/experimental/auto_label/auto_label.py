import argparse
import logging
import time
from pathlib import Path

import torch
import torchvision.transforms as T
import tqdm
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.inference import load_image, load_model, predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import (
    build_captions_and_token_span,
    create_positive_map_from_span,
)
from torchvision.ops import box_convert
from waffle_utils.file import io, search
from waffle_utils.log import initialize_logger

from waffle_hub.dataset import Dataset

initialize_logger("logs/auto_label.log")
logger = logging.getLogger(__name__)


class TimeChecker:
    def __init__(self, name: str = None):
        self.name = name
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not DEBUG:
            return
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
            logger.debug(f"{self.name}: {self.interval:.3f} sec")


class PostProcessGrounding(torch.nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, cat_list: list[str] = [], tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist
        )  # 80, 256. normed

        # build a mapping from label_id to pos_map
        id_map = {i: i + 1 for i in range(len(cat_list))}
        new_pos_map = torch.zeros((len(cat_list) + 1, 256))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        # pos map to logit
        prob_to_token = out_logits.sigmoid()  # bs, 100, 256
        pos_maps = self.positive_map.to(prob_to_token.device)
        # (bs, 100, 256) @ (91, 256).T -> (bs, 100, 91)
        prob_to_label = prob_to_token @ pos_maps.T

        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        # cxcywh to x1y1wh
        boxes = out_bbox
        boxes[:, :, :2] = boxes[:, :, :2] - boxes[:, :, 2:] / 2
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4)).cpu()

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cpu()
        boxes = boxes * scale_fct[:, None, :]

        return boxes, labels, scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="src/GroundingDINO_SwinT_OGC.py",
        help="path to config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        "-p",
        type=str,
        default="src/groundingdino_swint_ogc.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--source", "-s", type=str, required=True, help="path to image file or directory"
    )
    parser.add_argument("--text_prompt_file", "-t", type=str, required=True, help="text prompt file")
    parser.add_argument("--class_names_file", type=str, default=None, help="class name file")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--draw", action="store_true", help="draw bounding boxes")
    parser.add_argument("--recursive", action="store_true", help="load image recursively")

    parser.add_argument("--waffle_dataset_name", required=True, type=str, help="waffle dataset name")
    parser.add_argument(
        "--waffle_dataset_root_dir", default=None, type=str, help="waffle dataset root_dir"
    )

    parser.add_argument("--device", default="0", type=str, help="cuda device id or cpu")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    # cfg
    DEBUG = args.debug

    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model

    # text_prompt = args.text_prompt
    text_prompts = io.load_json(args.text_prompt_file)
    text_prompts = [text_prompt.lower() for text_prompt in text_prompts]
    text_prompt = " . ".join(text_prompts) + " ."

    # class mapper
    class_names = io.load_json(args.class_names_file) if args.class_names_file else text_prompts
    class2id = {name: i for i, name in enumerate(class_names)}
    id2class = {i: name for name, i in class2id.items()}

    prompt_to_class_name = {
        prompt: class_name for prompt, class_name in zip(text_prompts, class_names)
    }

    box_threshold = args.box_threshold
    text_threshold = args.text_threshold

    # build post processor
    cfg = SLConfig.fromfile(config_file)
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessGrounding(cat_list=text_prompts, tokenlizer=tokenlizer)

    # device
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"

    # directory
    source_dir = args.source
    output_dir = Path(args.output_dir)

    # load model
    model = load_model(config_file, checkpoint_path, device).to(device)

    # coco format
    coco = {
        "categories": [
            {"id": i + 1, "name": c, "supercategory": "object"} for i, c in id2class.items()
        ],
        "images": [],
        "annotations": [],
    }
    image_id = 1
    annotation_id = 1

    image_files = search.get_image_files(source_dir)
    for image_path in tqdm.tqdm(image_files, total=len(image_files)):

        with TimeChecker("load image"):
            image, image_tensor = load_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # run model
        h, w, _ = image.shape

        with TimeChecker("predict"):
            outputs = model(image_tensor, captions=[text_prompt])
        with TimeChecker("postprocess"):
            boxes, labels, scores = postprocessor(outputs, torch.Tensor([image.shape[:2]]))

        num_results = len(boxes)
        if len(boxes) == 0:
            continue

        # save result as coco format
        file_name = Path(image_path).relative_to(source_dir)
        coco["images"].append(
            {
                "id": int(image_id),
                "file_name": str(file_name),
                "width": int(w),
                "height": int(h),
            }
        )
        boxes = boxes[0].cpu()
        labels = labels[0].cpu()
        scores = scores[0].cpu()
        mask = scores > box_threshold

        for box, label_id, score in zip(boxes[mask], labels[mask], scores[mask]):
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_id.item(),
                    "bbox": box.tolist(),
                    "score": score.item(),
                }
            )
            annotation_id += 1
        image_id += 1

    # save coco format
    io.save_json(coco, output_dir / "coco.json", create_directory=True)

    # make waffle dataset
    dataset = Dataset.from_coco(
        name=args.waffle_dataset_name,
        task="object_detection",
        coco_file=output_dir / "coco.json",
        coco_root_dir=source_dir,
        root_dir=args.waffle_dataset_root_dir,
    )
    logger.info(f"Your waffle dataset has been saved to {dataset.dataset_dir}")

    if args.draw:
        dataset.draw_annotations()
