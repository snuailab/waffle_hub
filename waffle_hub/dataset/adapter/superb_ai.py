import logging
import warnings
from pathlib import Path
from typing import Union

import glob
import tqdm
from pycocotools.coco import COCO
from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.schema.fields import Annotation, Category, Image


def import_superb_ai(self, superb_project_json, superb_meta, superb_root_dir, cls_type):
    """
    Import coco dataset

    Args:
        superb_project_json (list[str]): List of Project json file
        superb_meta (list[str]): List of superb ai meta files root directories
        superb_root_dir (list[str]): List of superb ai meta files root directories
    """

    # cocos = [COCO(coco_file) for coco_file in coco_files]
    superb_project = io.load_json(superb_project_json)
    metas = glob.glob(f"{superb_meta}/**/*.json", recursive=True)

    superb_cat_id_to_waffle_cat_id = {}
    cats_id = 1
    for category in superb_project['object_detection']['object_classes']:
        if cls_type=='sub' and category.get('properties'):
            for cats in category['properties'][0]['options']:
                category_name = cats['name']
                superb_cat_id_to_waffle_cat_id[category_name] = cats_id
                category_super_name = category_name
                category_info = {
                    "id": cats_id,
                    "name": category_name,
                    "supercategory": category_super_name
                }
                self.add_categories([Category.from_dict({**category_info, "category_id": cats_id}, task=self.task)])

                cats_id +=1
                
        else:
            category_name = category['name']
            superb_cat_id_to_waffle_cat_id[category_name] = cats_id
            category_super_name = category_name
            category_info = {
                "id": cats_id,
                "name": category_name,
                "supercategory": category_super_name
            }
            self.add_categories([Category.from_dict({**category_info, "category_id": cats_id}, task=self.task)])

            cats_id +=1
    
    total_length = len(metas)
    logging.info(f"Importing superb ai dataset, Total Length: {total_length}")
    pgbar = tqdm.tqdm(total = total_length, desc="Importing SuperbAI dataset")

    image_id = 1
    annotation_id = 1

    for i, meta_path in enumerate(metas):
        image_ids = []
        meta = io.load_json(meta_path)
        file_name = meta['data_key']
        image_path = Path(superb_root_dir) / file_name[1:]

        if not image_path.exists():
            raise FileNotFoundError(f"{image_path} does not exist.")


        self.add_images(
            [Image.from_dict({"width": meta['image_info']['width'],
                              "height": meta['image_info']['height'],
                              "image_id": image_id,
                              "file_name": Path(file_name).name
                              })]
        )
        io.copy_file(image_path, self.raw_image_dir / Path(file_name).name, create_directory=True)

        anns_file = meta['label_path'][0]

        anno_data = io.load_json(f"{superb_root_dir}/{anns_file}")

        for label_info in anno_data['objects']:
            if label_info.get('properties'):
                if len(label_info['properties']) == 1 and len(label_info['properties'][0]['option_names']) == 1:
                    label_cls = label_info.get('properties')[0]['option_names'][0]
            else:
                label_cls = label_info['class_name']
            
            coord = label_info['annotation']['coord']

            annotation_dict = {
                "image_id": image_id,
                "annotation_id": annotation_id,
                "bbox": [coord['x'], coord['y'], coord['width'], coord['height']],
                'area': 0,
                'iscrowd': 0,
                "category_id": superb_cat_id_to_waffle_cat_id[label_cls]
            }
            self.add_annotations(
                [
                    Annotation.from_dict(
                        {
                            **annotation_dict
                        },
                        task=self.task,
                    )
                ]
            )
            annotation_id += 1
        image_ids.append(image_id)
        image_id += 1
        pgbar.update(1)
    
    pgbar.close()