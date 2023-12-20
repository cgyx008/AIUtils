import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


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

    # Read the txt
    with open(txt_path, 'r', encoding='utf-8') as f:
        labels = np.array([list(map(eval, line.split())) for line in f])
    if labels.size == 0:
        classes, boxes = [], []
    else:
        classes, boxes = labels[:, 0].astype(int), labels[:, 1:]
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

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
    for cls, box in zip(classes, boxes):
        cls_id = cls + cls_bias
        if cls_id > 2:
            continue
        cat = ('A', 'P', 'V')[cls_id]
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][cls_id]
        draw_rect_and_put_text(img, box, cat, color, 2)

    # Save the image
    if has_chinese:
        Image.fromarray(img[..., ::-1]).save(str(save_path))
    else:
        cv2.imwrite(str(save_path), img)


def vis_yolo_box(cwd, cls_bias=0, num_threads=8):
    cwd = Path(cwd)
    img_paths = sorted(cwd.glob('images/**/*.[jp][pn]g'))
    txt_paths = sorted(cwd.glob('labels/**/*.txt'))
    assert len(img_paths) == len(txt_paths)

    vis_dir = cwd / 'images_vis_labels'
    vis_dir.mkdir(exist_ok=True)
    save_paths = [vis_dir / img_path.relative_to(cwd / 'images')
                  for img_path in img_paths]
    # Create parent directories
    save_parents = sorted(list({str(p.parent) for p in save_paths}))
    for save_parent in tqdm(save_parents):
        Path(save_parent).mkdir(parents=True, exist_ok=True)

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


def verify_img(num_threads=8):
    img_dir = Path(r'U:\Animal\Public\kaggle\cats_and_dogs')
    img_paths = sorted(img_dir.glob('**/*.[jp][pn][g]'))

    def _verify_img(img_path):
        with Image.open(img_path) as im:
            im.verify()

    with ThreadPoolExecutor(num_threads) as executor:
        list(tqdm(executor.map(_verify_img, img_paths), total=len(img_paths)))


def main():
    verify_img()


if __name__ == '__main__':
    main()
