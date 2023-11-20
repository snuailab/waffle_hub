import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
import tqdm
from groundingdino.util.inference import load_image, load_model, predict
from torchvision.ops import box_convert
from waffle_utils.file import io, search
from waffle_utils.log import initialize_logger

from waffle_hub.dataset import Dataset

initialize_logger("logs/auto_label.log")


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
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model

    # text_prompt = args.text_prompt
    text_prompts = io.load_json(args.text_prompt_file)
    text_prompt = ".".join(text_prompts) + "."

    # class mapper
    class_names = io.load_json(args.class_names_file) if args.class_names_file else text_prompts
    class2id = {name: i for i, name in enumerate(class_names)}
    id2class = {i: name for name, i in class2id.items()}

    prompt_to_class_name = {
        prompt: class_name for prompt, class_name in zip(text_prompts, class_names)
    }

    box_threshold = args.box_threshold
    text_threshold = args.text_threshold

    # device
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"

    # directory
    source_dir = args.source
    output_dir = Path(args.output_dir)

    # load model
    model = load_model(config_file, checkpoint_path, device)

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

        image, image_tensor = load_image(image_path)

        # run model
        boxes, logits, phrases = predict(
            model,
            image_tensor,
            text_prompt,
            box_threshold,
            text_threshold,
            device,
        )

        h, w, _ = image.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = (
            box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy().astype(int).tolist()
        )
        classes_ids = [class2id[prompt_to_class_name[prompt]] for prompt in phrases]

        num_results = len(boxes)
        if len(boxes) == 0:
            continue

        # save result as coco format
        file_name = Path(image_path).relative_to(source_dir)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": str(file_name),
                "width": w,
                "height": h,
            }
        )
        for box, class_id in zip(boxes, classes_ids):
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": box,
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
    print(f"Your waffle dataset has been saved to {dataset.dataset_dir}")

    if args.draw:
        dataset.draw_annotations()
