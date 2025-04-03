import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

if __name__ == '__main__':
    import sys
    sys.path.insert(0, Path(__file__).parents[1].absolute().as_posix())
from data.dataset import get_img_txt_xml


def get_coco_json_format(json_path):
    """
    COCO json format:
    json_data = {
        'info': {'description': 'COCO 2017 Dataset',
                 'url': 'http://cocodataset.org',
                 'version': '1.0',
                 'year': 2017,
                 'contributor': 'COCO Consortium',
                 'date_created': '2017/09/01'},
        'licenses': [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
                      'id': 1,
                      'name': 'Attribution-NonCommercial-ShareAlike License'},
                     ...],
        'images': [{'licence': 4,
                    'file_name': '000000397133.jpg',
                    'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg',
                    'height': 427,
                    'width': 640,
                    'date_captured': '2013-11-14 17:02:52',
                    'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg',
                    'id': 397133},
                   ...],
        'annotations': [{'segmentation': [[510.66, 423.01, ..., 423.01]],
                         'area': 702.1057499999998,
                         'iscrowd': 0,
                         'image_id': 289343,  # same with 'id' in 'images'
                         'bbox': [473.07, 395.93, 38.65, 28.67],  # [xmin, ymin, w, h]
                         'category_id': 18,
                         'id': 1768},  # annotation id
                        ...],
        'categories': [{'supercategory': 'person', 'id': 1, 'name': 'person'}, ...]
    }

    """
    # json_path = r'W:\ganhao\Public\COCO\annotations\instances_val2017.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    assert json_data


def get_images_and_annotations(img_id_and_txt, name2img):
    """
    Get images and annotations in COCO json data format.
    Args:
        img_id_and_txt (Path): image index and txt path
        name2img (dict): {name: img_path}

    Returns:
        tuple[dict, list]: (images, annotations)
    """
    img_id, txt = img_id_and_txt
    img_path = name2img.get(txt.stem, None)
    if not img_path:
        return {}, [{}]
    with Image.open(img_path) as im:
        w, h = im.size

    # image dict
    image = {'license': 1, 'file_name': Path(img_path).name, 'coco_url': '',
             'height': h, 'width': w, 'date_captured': '', 'flickr_url': '',
             'id': img_id}

    # annotation list
    with open(txt, 'r', encoding='utf-8') as f:
        anns = [list(map(eval, line.split())) for line in f]
    anns = np.asarray(anns)
    if not anns.size:
        return image, [{}]
    class_ids, boxes = anns[:, 0], anns[:, 1:5]
    class_ids = class_ids.astype(int).tolist()
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes = boxes.tolist()
    annotations = [{'segmentation': [], 'area': box[2] * box[3], 'iscrowd': 0,
                    'image_id': img_id, 'bbox': box, 'category_id': cid,
                    'id': aid}
                   for aid, (box, cid) in enumerate(zip(boxes, class_ids))]

    return image, annotations


def txt2json(img_paths, txt_paths, json_path,
              categories=('animal', 'person', 'vehicle', 'package')):
    """Transform YOLO format txts to COCO format json.
    Args:
        img_paths list(str | Path): images paths.
        txt_paths list(str | Path): txt paths.
        json_path (str | Path): Save json path.
        categories (tuple[str] | list[str]): class names.
    """
    json_data = {
        'info': {'description': 'WDOV Dataset', 'url': '',
                 'version': '1.0', 'year': datetime.now().year,
                 'contributor': 'Reolink Algorithm',
                 'date_created': datetime.now().strftime('%Y/%m/%d')},
        'licenses': [{'url': '', 'id': 1, 'name': 'Reolink License'}],
        'images': [],
        'annotations': [],
        'categories': [{'supercategory': c, 'id': i, 'name': c}
                       for i, c in enumerate(categories)]
    }

    name2img = {p.stem: p for p in img_paths}

    with ThreadPoolExecutor(8) as exe:
        data = list(tqdm(
            exe.map(partial(get_images_and_annotations, name2img=name2img),
                    enumerate(txt_paths)),
            total=len(txt_paths)
        ))
    imgs, anns = zip(*data)
    imgs = [f for f in imgs if f]
    img_ids_map = {img['id']: i for i, img in enumerate(imgs)}
    for i, img in enumerate(imgs):
        img['id'] = i
    anns = [a for al in anns for a in al if a]
    anns.sort(key=lambda a: (a['image_id'], a['id']))
    for i, a in enumerate(anns):
        a['image_id'] = img_ids_map[a['image_id']]
        a['id'] = i
    json_data['images'], json_data['annotations'] = imgs, anns
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)


def json2txts(json_path):
    """Transform COCO json to YOLO format txts.
    Args:
        json_path (str | Path): COCO json path.
    """
    json_path = Path(json_path)
    label_dir = json_path.parent / 'json2labels'
    os.umask(0)
    label_dir.mkdir(parents=True, exist_ok=True)

    anns = COCO(json_path)
    for img_id, img_dict in tqdm(anns.imgs.items()):
        file_name = img_dict['file_name']
        img_w, img_h = img_dict['width'], img_dict['height']
        img_anns = anns.imgToAnns[img_id]
        txt_path = label_dir / file_name.replace('.jpg', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for ann in img_anns:
                category_id = ann['category_id']
                xmin, ymin, w, h = ann['bbox']
                xc = (xmin + w / 2) / img_w
                yc = (ymin + h / 2) / img_h
                w /= img_w
                h /= img_h
                f.write(f'{category_id} {xc} {yc} {w} {h}\n')

    category_txt = json_path.parent / 'categories_in_json.txt'
    with open(category_txt, 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(anns.cats))):
            f.write(f"{anns.cats[i]['name']}\n")


def generate_json_from_ov_dino_labels(root):
    # root = Path('/home/ganhao/data/ppvpd/open_vocabulary/20240817')
    txt_paths = sorted(root.glob('labels/*.txt'))
    img_paths = [get_img_txt_xml(t)[0] for t in txt_paths]
    json_path = root / 'ppvpd_20240817_open_vocabulary_annotations_by_spacy.json'
    with open(root / 'categories_from_ovd_by_spacy.txt', 'r', encoding='utf-8') as f:
        categories = [c.strip() for c in f]
    txt2json(img_paths, txt_paths, json_path, categories)


def make_oiv7_json(root):
    # root = Path('/mnt/28Server')
    img_paths = sorted(root.glob('images/**/*.jpg'))  # 1743042
    txt_paths = [get_img_txt_xml(img_path)[1] for img_path in img_paths]
    json_path = root / 'annotations.json'
    with open(root / 'category.txt', 'r', encoding='utf-8') as f:
        categories = [line.strip().rsplit(' ', maxsplit=1)[0] for line in f]
    print(f'{len(categories) = }')
    print(f'{categories[0] = }')
    txt2json(img_paths, txt_paths, json_path, categories)


def main():
    ...


if __name__ == '__main__':
    main()
