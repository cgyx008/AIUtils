from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
from tqdm import tqdm

from image.image_utils import draw_rect_and_put_text


def xcycwh2xyxy(box: np.ndarray):
    box_cp = box.copy()
    box_cp[:, 0] -= box_cp[:, 2] / 2
    box_cp[:, 1] -= box_cp[:, 3] / 2
    box_cp[:, 2] += box_cp[:, 0]
    box_cp[:, 3] += box_cp[:, 1]
    return box_cp


def cmp_label_txt_and_pred_txt(label_txt, pred_txt, num_classes=3):
    """
    Match labels and predictions
    Args:
        label_txt (str | Path): [class_id, xc, yc, w, h], normed
        pred_txt (str | Path): [class_id, xc, yc, w, h], normed
        num_classes (int): the number of classes

    Returns:

    """
    tp, fp, fn = [], [], []

    if Path(label_txt).exists():
        with open(label_txt) as f:
            labels = np.array([list(map(eval, line.split())) for line in f])
    else:
        labels = np.array([])

    if Path(pred_txt).exists():
        with open(pred_txt) as f:
            preds = np.array([list(map(eval, line.split())) for line in f])
    else:
        preds = np.array([])

    # xcycwh -> xyxy
    if labels.size:
        labels[:, 1:5] = xcycwh2xyxy(labels[:, 1:5])
    if preds.size:
        preds[:, 1:5] = xcycwh2xyxy(preds[:, 1:5])

    if not labels.size and not preds.size:
        return tp, fp, fn
    elif not labels.size:
        fp.append(preds)
        return tp, fp, fn
    elif not preds.size:
        fn.append(labels[:, 1:5])
        return tp, fp, fn

    for i in range(num_classes):
        labels_i = labels[labels[:, 0] == i][:, 1:5]
        preds_i = preds[preds[:, 0] == i][:, 1:5]

        if not labels_i.size and not preds_i.size:
            continue
        elif not labels_i.size:
            fp.append(preds[preds[:, 0] == i])
            continue
        elif not preds_i.size:
            fn.append(labels_i)
            continue

        # xcycwh -> x1y1x2y2
        # labels_i = xcycwh_2_x1y1x2y2(labels_i)
        # preds_i = xcycwh_2_x1y1x2y2(preds_i)

        iou = box_iou(torch.tensor(labels_i), torch.tensor(preds_i))
        iou = iou * (iou > 0.45)
        iou = iou.numpy()

        labels_ids, preds_ids = linear_sum_assignment(iou, maximize=True)

        for j in range(labels_i.shape[0]):
            if j not in labels_ids:
                fn.append(labels_i[j][None, :])
        for j in range(preds_i.shape[0]):
            if j not in preds_ids:
                fp.append(preds[preds[:, 0] == i][j][None, :])
        for label_idx, preds_idx in zip(labels_ids, preds_ids):
            if iou[label_idx, preds_idx] > 0:
                tp.append(labels_i[label_idx][None, :])
            else:
                fn.append(labels_i[label_idx][None, :])
                fp.append(preds[preds[:, 0] == i][preds_idx][None, :])

    return tp, fp, fn


def _save_fp_and_fn(line, pred_dir, save_dir, num_classes=3):
    line = line.strip()
    img_path = line.replace('/home/kemove/218Algo', 'W:')
    img_path = img_path.replace('/home/kemove', 'Z:')
    label_txt = img_path.replace('/images/', '/labels/')
    label_txt = Path(label_txt).with_suffix('.txt')
    pred_txt = Path(pred_dir) / label_txt.name

    tp, fp, fn = cmp_label_txt_and_pred_txt(label_txt, pred_txt, num_classes)

    if not fp and not fn:
        return

    tp = np.concatenate(tp, axis=0) if tp else np.array([])
    fp = np.concatenate(fp, axis=0) if fp else np.array([])
    fn = np.concatenate(fn, axis=0) if fn else np.array([])

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    for box in tp:
        box[[0, 2]] *= w
        box[[1, 3]] *= h
        box = box.astype(int)
        draw_rect_and_put_text(img, box, 'TP', (0, 255, 0), 2)
    for box in fp:
        class_id = str(int(box[0]))
        box = box[1:5]
        box[[0, 2]] *= w
        box[[1, 3]] *= h
        box = box.astype(int)
        draw_rect_and_put_text(img, box, class_id, (0, 0, 255), 2)
    for box in fn:
        box[[0, 2]] *= w
        box[[1, 3]] *= h
        box = box.astype(int)
        draw_rect_and_put_text(img, box, 'FN', (255, 0, 0), 2)

    if fp.size:
        cv2.imwrite(fr'{save_dir}\FP\{Path(img_path).name}', img)
    if fn.size:
        cv2.imwrite(fr'{save_dir}\FN\{Path(img_path).name}', img)


def save_fp_and_fn():
    """Save FP and FN images"""
    txt = r'Z:\8TSSD\ganhao\data\fepvd\v05\test.txt'
    with open(txt) as f:
        lines = f.readlines()

    pred_dir = r'Z:\8TSSD\ganhao\projects\ultralytics\runs\detect\fepvd\predict\fepvd_v05_000_epochs_300_test\labels'
    save_dir = r'T:\Working\v05\clean_test'
    pred_dirs = [pred_dir] * len(lines)
    save_dirs = [save_dir] * len(lines)
    (Path(save_dir) / 'fp').mkdir(parents=True, exist_ok=True)
    (Path(save_dir) / 'fn').mkdir(parents=True, exist_ok=True)
    num_classes_list = [2] * len(lines)

    with ThreadPoolExecutor(8) as executor:
        list(tqdm(executor.map(_save_fp_and_fn,
                               lines, pred_dirs, save_dirs, num_classes_list),
                  total=len(lines)))


def _rm_duplicate_lines_in_txts(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    write_lines, unique_lines = [], set()
    for line in lines:
        if line not in unique_lines:
            unique_lines.add(line)
            write_lines.append(line)
    if len(lines) == len(write_lines):
        return
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(write_lines)


def rm_duplicate_lines_in_txts():
    cwd = Path(r'Z:\2TSSD\ganhao\Projects\ultralytics\runs\detect\predict\wdv04_002_remove_coco_val\labels')
    txt_paths = sorted(cwd.glob('*.txt'))

    with ThreadPoolExecutor(8) as executor:
        list(tqdm(executor.map(_rm_duplicate_lines_in_txts,txt_paths),
                  total=len(txt_paths)))


def main():
    # cmp_label_txt_and_pred_txt(
    #     r'W:\ganhao\AD\wd\v04\labels\000\5858c175-23d2-11e8-a6a3-ec086b02610b.txt',
    #     r'Z:\2TSSD\ganhao\Projects\ultralytics\runs\detect\predict\wdv04_000_val\labels\5858c175-23d2-11e8-a6a3-ec086b02610b.txt',
    #     num_classes=3)
    save_fp_and_fn()


if __name__ == '__main__':
    main()
