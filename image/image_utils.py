import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

if __name__ == '__main__':
    import sys
    sys.path.insert(0, Path(__file__).parents[1].as_posix())

from data.dataset import get_img_txt_xml
from data.txt2xml import read_xml
from file.ops import create_parent_dirs


def draw_rect_and_put_text(img, box, text, color=(0, 0, 255), box_thickness=1,
                           font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6,
                           line_type=cv2.LINE_AA):
    # bounding-box
    h, w, _ = img.shape
    box = list(map(int, box))
    pt1 = (max(box[0], 0), max(box[1], 0))
    pt2 = (min(box[2], w - 1), min(box[3], h - 1))
    cv2.rectangle(img, pt1, pt2, color, box_thickness, lineType=line_type)

    # text-box
    tbox_color = color
    text_size = cv2.getTextSize(text, font, font_scale, box_thickness * 2)
    if pt1[1] - text_size[0][1] < 0:
        tbox_pt1 = (pt1[0], pt1[1])
        tbox_pt2 = (pt1[0] + text_size[0][0], pt1[1] + text_size[0][1])
    else:
        tbox_pt1 = (pt1[0], pt1[1] - text_size[0][1])
        tbox_pt2 = (pt1[0] + text_size[0][0], pt1[1])
    cv2.rectangle(img, tbox_pt1, tbox_pt2, tbox_color, -1, lineType=line_type)

    # text
    if pt1[1] - text_size[0][1] < 0:
        text_pt = (pt1[0], pt1[1] + text_size[0][1])
    else:
        text_pt = pt1
    tcolor = (255, 255, 255)
    cv2.putText(img, text, text_pt, font, font_scale, tcolor, box_thickness * 2,
                lineType=line_type)
    return img


def keep_wh_resize(src, dsize):
    """
    Keep w/h to resize the image
    Args:
        src (np.ndarray): the image
        dsize (tuple): (width, height)
    """
    dst_r = dsize[0] / dsize[1]
    src_h, src_w = src.shape[:2]
    src_r = src_w / src_h
    if src_r > dst_r:
        new_w, new_h = dsize[0], int(dsize[0] / src_r)
    else:
        new_w, new_h = int(dsize[1] * src_r), dsize[1]
    dst = cv2.resize(src, (new_w, new_h))

    # Pad
    pad_w, pad_h = dsize[0] - new_w, dsize[1] - new_h
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    padded_img = cv2.copyMakeBorder(dst, pad_t, pad_b, pad_l, pad_r,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded_img


def cv2_imshow(img_path):
    img = cv2.imread(str(img_path))
    img = keep_wh_resize(img, (960, 540))
    cv2.namedWindow('img')
    cv2.moveWindow('img', 0, 0)
    cv2.imshow('img', img)

    key = cv2.waitKey(0)
    if key == ord('j'):
        ...
    elif key == ord('k'):
        ...
    elif key == ord('l'):
        ...
    elif key == ord(';'):
        ...
    else:
        ...


def vis_an_image_and_boxes(img_path, txt_path, save_path, cls_bias=0):
    if save_path.exists():
        return
    assert img_path.stem == txt_path.stem
    if not Path(txt_path).exists():
        return

    # Read the txt
    with open(txt_path, 'r', encoding='utf-8') as f:
        labels = np.array([list(map(eval, line.split())) for line in f])
    if labels.size == 0:
        classes, boxes, confs = [], [], []
    else:
        classes, boxes = labels[:, 0].astype(int), labels[:, 1:5]
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        confs = labels[:, 5] if labels.shape[1] == 6 else [-1] * len(labels)

    # Read the image
    img = cv2.imread(str(img_path))
    has_chinese = False
    if img is None:
        with Image.open(str(img_path)) as img:
            img = img.convert('RGB')
        img = np.ascontiguousarray(np.array(img)[..., ::-1])
        has_chinese = True
        # print(f'The image is None! {img_path}')
        # return

    # Draw boxes
    h, w = img.shape[:2]
    if isinstance(boxes, np.ndarray):
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
    for cls, box, conf in zip(classes, boxes, confs):
        cls_id = cls + cls_bias
        if cls_id > 2:
            continue
        cat = ('A', 'P', 'V')[cls_id]
        text = f'{cat} {conf:.2f}' if conf >= 0 else cat
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][cls_id]
        draw_rect_and_put_text(img, box, text, color, 2)

    # Save the image
    if has_chinese:
        Image.fromarray(img[..., ::-1]).save(str(save_path))
    else:
        cv2.imwrite(str(save_path), img)


def vis_yolo_box(cwd, save_dir=None, cls_bias=0, num_threads=8):
    cwd = Path(cwd)
    img_paths = sorted(cwd.glob('**/images/**/*.[jp][pn]g'))
    txt_paths = [get_img_txt_xml(p)[1] for p in tqdm(img_paths)]

    if save_dir is None:
        save_paths = [
            Path(str(p).replace('images', 'images_vis_labels'))
            for p in img_paths
        ]
    else:
        save_dir = Path(save_dir)
        save_paths = [save_dir / p.relative_to(cwd) for p in img_paths]

    # Create parent directories
    create_parent_dirs(save_paths)

    cls_bias = [cls_bias] * len(img_paths)

    with ThreadPoolExecutor(num_threads) as executor:
        list(tqdm(
            executor.map(vis_an_image_and_boxes,
                         img_paths, txt_paths, save_paths, cls_bias),
            total=len(img_paths)
        ))


def vis_mmap(mmap_path):
    mmap_path = Path(mmap_path)
    shape = re.findall(r'\d+', mmap_path.stem)
    shape = tuple(map(int, shape))
    imgs = np.memmap(mmap_path, mode='r', shape=shape)
    for idx in [0, 1, -1]:
        img = imgs[idx]
        img_path = mmap_path.parent / f'{mmap_path.stem}_{idx}.jpg'
        cv2.imwrite(str(img_path), img)


def verify_imgs(num_threads=8):
    img_dir = Path(r'F:\data\AD\nz_trailcams\images')
    img_paths = sorted(img_dir.glob('**/*.[jp][pn][g]'))

    def verify_img(img_path):
        try:
            with Image.open(img_path) as im:
                im.verify()
        except UnidentifiedImageError as e:
            print(e, img_path)

    with ThreadPoolExecutor(num_threads) as executor:
        list(tqdm(executor.map(verify_img, img_paths), total=len(img_paths)))


def gen_img_id(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path)
    if img is None:
        with Image.open(img_path) as img:
            img = np.array(img)
    h, w = img.shape[:2]
    s, v = img.sum(), img.var()
    return f'{w}_{h}_{s}_{v}'


def get_img_ids_in_txt(txt_path):
    ids_txt = Path(txt_path).with_stem(f'{Path(txt_path).stem}_ids')
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    img_paths = [line.strip() for line in lines]
    for img_path in tqdm(img_paths):
        img_id = gen_img_id(img_path)
        with open(ids_txt, 'a', encoding='utf-8') as f:
            f.write(f'{img_path} {img_id}\n')


def get_img_ids_in_dir(img_dir, num_threads=8):
    img_dir = Path(img_dir)
    ids_txt = img_dir / f'{img_dir.stem}_img_ids.txt'
    img_paths = sorted(img_dir.glob('**/images/**/*.jpg'))

    with ThreadPoolExecutor(num_threads) as executor:
        img_ids = list(
            tqdm(executor.map(gen_img_id, img_paths), total=len(img_paths)))

    lines = [f'{p.as_posix()} {img_id}\n'
             for p, img_id in zip(tqdm(img_paths), img_ids)]

    with open(ids_txt, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def write_down_dup_imgs(txt_path):
    """
    Write down duplicate images according to the image id.
    Args:
        txt_path (str | Path): image id txt
    """
    txt_path = Path(txt_path)
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    id2img = {}
    for line in tqdm(lines):
        img_path, img_id = line.strip().rsplit(' ', 1)
        if img_id not in id2img:
            id2img[img_id] = [img_path]
        else:
            id2img[img_id].append(img_path)

    id2dupimg = {k: v for k, v in id2img.items() if len(v) > 1}
    dup_json = txt_path.with_name(f'{txt_path.stem}_dup_imgs.json')
    with open(dup_json, 'w', encoding='utf-8') as f:
        json.dump(id2dupimg, f, indent=4)


def vis_dup_imgs(json_path):
    json_path = Path(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    for img_id, img_paths in tqdm(json_data.items()):
        save_dir = json_path.parent / 'useless_images/duplicate_images' / img_id
        os.umask(0)
        save_dir.mkdir(parents=True, exist_ok=True)
        for img_path in img_paths:
            shutil.copy2(img_path, save_dir)


def rm_dup_imgs(json_path):
    json_path = Path(json_path)
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    dup_imgs = []
    for dup_img_pairs in tqdm(json_data.values()):
        picked, pick_idx = False, -1
        for i, img in enumerate(dup_img_pairs):
            if 'reolink_user' in img:
                picked, pick_idx = True, i
                break
        dup_img_pairs.pop(picked and pick_idx or 0)
        dup_imgs.extend(dup_img_pairs)

    rm_path = json_path.with_stem(f'{json_path.stem}_rm')
    with open(rm_path, 'w', encoding='utf-8') as f:
        json.dump(dup_imgs, f, indent=4)

    for p in dup_imgs:
        Path(p).unlink()

def collate_imgs_without_xml(root):
    root = Path(root)

    no_xml_dir = root / 'useless_images/images_without_xml'
    no_xml_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(root.glob('images/**/*.jpg'))
    for img_path in tqdm(img_paths):
        xml_path = get_img_txt_xml(img_path)[2]

        if not xml_path.exists():
            shutil.copy2(img_path, no_xml_dir)


def collate_imgs_with_difficult_obj(root):
    root = Path(root)

    difficult_dir = root / 'useless_images/images_with_difficult_object'
    difficult_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(root.glob('images/**/*.jpg'))
    for img_path in tqdm(img_paths):
        xml_path = get_img_txt_xml(img_path)[2]

        if read_xml(xml_path)['has_difficult']:
            shutil.copy2(img_path, difficult_dir)


def collate_useless_imgs(root):
    collate_imgs_without_xml(root)
    collate_imgs_with_difficult_obj(root)


def rm_useless_images(root):
    root = Path(root)
    # 1. Remove images with difficult objects
    useless_imgs = sorted(root.glob('useless_images/*/*.jpg'))

    # 2. Remove duplicate images
    json_path = sorted(root.glob('*_dup_imgs.json'))[0]
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    # dup_imgs = [Path(p) for d in json_data.values() for p in d[1:]]

    dup_imgs = []
    for dup_img_pairs in tqdm(json_data.values()):
        picked, pick_idx = False, -1
        for i, img in enumerate(dup_img_pairs):
            if 'reolink_user' in img:
                picked, pick_idx = True, i
                break
        dup_img_pairs.pop(picked and pick_idx or 0)
        dup_imgs.extend(dup_img_pairs)

    # 3. Combine 2 lists
    rm_img_stems = {p.stem for p in useless_imgs + dup_imgs}

    # 4. Remove images
    img_paths = sorted(root.glob('images/**/*.jpg'))
    for img_path in tqdm(img_paths):
        if img_path.stem in rm_img_stems:
            img_path.unlink(missing_ok=True)


def main():
    vis_yolo_box(
        r'G:\data\wd\v006'
    )


if __name__ == '__main__':
    main()
