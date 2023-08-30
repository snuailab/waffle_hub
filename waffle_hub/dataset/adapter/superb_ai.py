import logging
import warnings
from pathlib import Path
from typing import Union

import glob
import tqdm
from waffle_utils.file import io

from waffle_hub import TaskType
from waffle_hub.schema.fields import Annotation, Category, Image

from collections import defaultdict
from datetime import datetime

import random

def generate_random_color_code(num_colors):
    color_codes = []
    for _ in range(num_colors):
        # 무작위 RGB 값을 생성합니다.
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        
        # RGB 값을 16진수로 변환하여 색상 코드를 생성합니다.
        color_code = "#{:02X}{:02X}{:02X}".format(red, green, blue)
        
        color_codes.append(color_code)
    
    return color_codes

def export_superb_ai(self, export_dir: Union[str, Path]) -> str:
    """Export dataset to SuperbAI format
    Args:
        export_dir (Path): Path to export directory
    
    Returns:
        str: Path to export directory
    """
    logging.info(f"Exporting.. SuperAi Dataset.")
    export_dir = Path(export_dir)
    io.make_directory(export_dir)

    image_dir = export_dir / "images"
    meta_dir  = export_dir / 'meta'
    label_dir = export_dir / 'labels'

    io.make_directory(image_dir)
    io.make_directory(meta_dir)
    io.make_directory(label_dir)

    num_colors = len(self.get_categories())
    colors = generate_random_color_code(num_colors)

    super_cats = defaultdict(list)
    [super_cats[category.supercategory].append(category.name) for i, category in enumerate(self.get_categories())]
    
    object_classes = []
    check_lst = []
    for i, category in enumerate(self.get_categories()):
        if category.supercategory not in check_lst:
            check_lst.append(category.supercategory)
            object_classes.append({
                "id": category.category_id,
                "name": category.supercategory,
                "color": colors[i],
                "properties": [],
                "constraints": {},
                "ai_class_map": [],
                "annotation_type": "box" 
            })

    options = []
    for object in object_classes:
        if len(super_cats[object['name']]) == 1:
            object['properties'] = []
        else:
            for option_name in super_cats[object['name']]:
                options.append(
                        {
                            "id": category.category_id,
                            "name": option_name
                        },
                    )
            object['properties']= [ {
                    "id": category.category_id,
                    "name": f"{object['name']}_Type",
                    "type": "checkbox",
                    "options": options,
                    "required": True,
                    "description": "",
                    "render_value": False,
                    "default_value": []
                }
            ]

    project = {
        'type': "image-siesta", # Unknown
        'version': '0.6.5', # Unknown
        'data-type': 'image', # Unknown
        'categorization':{
            "properties": []
            },
        TaskType.OBJECT_DETECTION.name.lower() : {
            'keypoints': [],
            'object_groups': [],
            "object_classes": object_classes,
            "annotation_types":[
                "box"
                ]
            }
        }
    io.save_json(project, str(export_dir / 'project.json'), create_directory=True)


    for image in self.get_images():
        image_path = self.raw_image_dir / image.file_name
        image_dst_path = image_dir / image.file_name
        io.copy_file(image_path, image_dst_path, create_directory=True)

        meta_path = f'{meta_dir}/{image.file_name}.json'
        label_path = f'{label_dir}/{image.image_id}.json'

        meta = {
            'data_key': f"images/{image.file_name}",
            'dataset': self.get_dataset_info().name,
            'image_info': {
                'width': image.width,
                'height': image.height
            },
            "label_id": image.image_id,
            "label_path": [
                f"labels/{image.image_id}.json"
                ],
            "last_updated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tags": [],
            "work_assignee": "research@snuailab.ai",
            "status": "Waffle_dataset to Sueprb AI Dataset"
        }
        io.save_json(meta ,Path(meta_dir) / meta_path)

        label_lst = []
        for i, annotation in enumerate(self.get_annotations(image.image_id)):
            supercategory = self.get_categories([annotation.category_id])[0].supercategory
            category = self.get_categories([annotation.category_id])[0].name

            if supercategory == category:
                properties = []
            else:
                properties = [
                        {
                            "type": "checkbox",
                            "property_id": annotation.category_id,
                            "property_name": f'{supercategory}_Type',
                            "option_names": [
                                self.get_categories([annotation.category_id])[0].name
                            ]
                        }
                    ]

            label_lst.append(
                    {
                        "id": annotation.annotation_id,
                        "class_id": annotation.category_id,
                        "class_name": self.get_categories([annotation.category_id])[0].supercategory,
                        "annotation_type": "box",
                        "annotation": {
                            "coord": {
                                "x": annotation.bbox[0],
                                "y": annotation.bbox[1],
                                "width": annotation.bbox[2],
                                "height": annotation.bbox[3]
                            },
                            "meta": {
                                "z_index": i,
                                "visible": True,
                                "alpha": 1,
                                "color": colors[annotation.category_id-1]
                            }

                        },
                        "properties": properties
                    }
                )
        label_json = {'objects': label_lst}
        io.save_json(label_json, label_path, create_directory=True)

    logging.info(f"Exported SuperAi Dataset. Export_dir Path: {export_dir}")

    return str(export_dir)
    

def import_superb_ai(self, superb_project_json, superb_meta, superb_root_dir):
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
        supercategory_name = category['name']
        if category.get('properties'):
            for cats in category['properties'][0]['options']:
                category_name = cats['name']
                superb_cat_id_to_waffle_cat_id[category_name] = cats_id
                category_info = {
                    "id": cats_id,
                    "supercategory": supercategory_name,
                    "name": category_name
                }
                self.add_categories([Category.from_dict({**category_info, "category_id": cats_id}, task=self.task)])

                cats_id +=1
                
        else:
            category_name = supercategory_name
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

    for meta_path in metas:
        image_ids = []
        meta = io.load_json(meta_path)
        file_name = meta['data_key']
        image_path = Path(superb_root_dir) / (file_name[1:] if file_name[0] == '/' else file_name)

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