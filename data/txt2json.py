import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_coco_json_format():
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
                         'bbox': [473.07, 395.93, 38.65, 28.67],
                         'category_id': 18,
                         'id': 1768},  # annotation id
                        ...],
        'categories': [{'supercategory': 'person', 'id': 1, 'name': 'person'}, ...]
    }

    """
    json_path = r'W:\ganhao\Public\COCO\annotations\instances_val2017.json'
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
    image = {'license': 1, 'file_name': str(img_path), 'coco_url': '',
             'height': h, 'width': w, 'date_captured': '', 'flickr_url': '',
             'id': img_id}

    # annotation list
    with open(txt, 'r', encoding='utf-8') as f:
        anns = [list(map(eval, line.split())) for line in f]
    anns = np.asarray(anns)
    if not anns.size:
        return image, [{}]
    class_ids, boxes = anns[:, 0], anns[:, 1:5]
    class_ids = class_ids.astype(np.int).tolist()
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


def txt2json(txt_dir, img_dir, json_path, classes=('Pet', 'Person', 'Vehicle')):
    """Transform YOLO format txts to COCO format json.

    Args:
        txt_dir (str | Path): txt directory.
        img_dir (str | Path): images directory.
        json_path (str | Path): Save json path.
        classes (tuple[str]): class names.
    """
    json_data = {
        'info': {'description': 'PPVD 2023 Dataset', 'url': '',
                 'version': '1.0', 'year': 2023,
                 'contributor': 'Reolink Algorithm Group Members',
                 'date_created': '2023/02/03'},
        'licenses': [{'url': '', 'id': 1, 'name': 'Reolink License'}],
        'images': [],
        'annotations': [],
        'categories': [{'supercategory': c, 'id': i, 'name': c}
                       for i, c in enumerate(classes)]
    }

    print('Searching images...')
    imgs = sorted(list(Path(img_dir).glob('**/*.[jp][pn]g')))
    name2img = {p.stem: p for p in imgs}
    # assert len(imgs) == len(name2img)

    print('Searching txts...')
    txts = sorted(list(Path(txt_dir).glob('**/*.txt')))

    with ThreadPoolExecutor(8) as exe:
        data = list(tqdm(
            exe.map(partial(get_images_and_annotations, name2img=name2img),
                    enumerate(txts)),
            total=len(txts)
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
    assert 1


def main():
    txt2json(
        'W:/liaojianhui/PPVD/20230203/labels',
        'W:/liaojianhui/PPVD',
        'W:/ganhao/PPVD/20230203/labels.json'
    )


if __name__ == '__main__':
    main()
