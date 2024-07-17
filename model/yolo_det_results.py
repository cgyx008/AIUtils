import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
from tqdm import tqdm

if __name__ == '__main__':
    import sys
    sys.path.insert(0, Path(__file__).parents[1].as_posix())
from file.ops import create_parent_dirs, cp
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
    has_chinese = False
    if img is None:
        has_chinese = True
        with Image.open(img_path) as img:
            img = np.ascontiguousarray(np.array(img)[..., ::-1])
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
        if has_chinese:
            Image.fromarray(img[..., ::-1]).save(fr'{save_dir}\fp\{Path(img_path).name}')
        else:
            cv2.imwrite(fr'{save_dir}\fp\{Path(img_path).name}', img)
    if fn.size:
        if has_chinese:
            Image.fromarray(img[..., ::-1]).save(fr'{save_dir}\fn\{Path(img_path).name}')
        else:
            cv2.imwrite(fr'{save_dir}\fn\{Path(img_path).name}', img)


def save_fp_and_fn(txt, pred_dir, save_dir, num_workers=8):
    """
    Save FP and FN images
    Args:
        txt (str): source txt file
        pred_dir (str): yolo labels directory
        save_dir (str): save directory
        num_workers (int): number of workers

    Examples:
        >>> txt = 'Z:/8TSSD/ganhao/data/wd/v04/trainval.txt'
        >>> pred_dir = 'Z:/8TSSD/ganhao/projects/ultralytics/runs/detect/wd/predict/wd_v05_000_trainval/labels'
        >>> save_dir = 'G:/data/wd/working/clean_train_val'
        >>> save_fp_and_fn(txt, pred_dir, save_dir)
    """
    with open(txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pred_dirs = [pred_dir] * len(lines)
    save_dirs = [save_dir] * len(lines)
    (Path(save_dir) / 'fp').mkdir(parents=True, exist_ok=True)
    (Path(save_dir) / 'fn').mkdir(parents=True, exist_ok=True)
    num_classes_list = [3] * len(lines)

    with ThreadPoolExecutor(num_workers) as executor:
        list(tqdm(executor.map(_save_fp_and_fn,
                               lines, pred_dirs, save_dirs, num_classes_list),
                  total=len(lines)))


def cp_fp_fn_labels():
    save_dir = Path(r'G:\data\wd\working\clean_train_val')
    fp = sorted(save_dir.glob('fp/*.[jp][pn]g'))
    fn = sorted(save_dir.glob('fn/*.[jp][pn]g'))
    stems = {p.stem for p in (fp + fn)}

    txt = r'Z:\8TSSD\ganhao\data\wd\v04\trainval.txt'
    with open(txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    label_dir = save_dir / 'labels'
    label_dir.mkdir(exist_ok=True)

    for line in tqdm(lines):
        line = line.strip()
        img_path = line.replace('/home/kemove/218Algo', 'W:')
        img_path = img_path.replace('/home/kemove', 'Z:')
        txt_path = img_path.replace('/images/', '/labels/')
        txt_path = Path(txt_path).with_suffix('.txt')
        if txt_path.stem in stems:
            shutil.copy2(txt_path, label_dir)


def _rm_duplicate_lines_in_txts(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
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


def mv_predicted_imgs_in_a_dir(root):
    root = Path(root)
    src_paths = sorted(root.glob('*.txt'))
    dst_paths = [p.parent / 'labels_iqa' /p.name for p in tqdm(src_paths)]
    create_parent_dirs(dst_paths)
    for src, dst in zip(tqdm(src_paths), dst_paths):
        src.rename(dst)


def mv_pred_label_dir():
    """
    Move predicted label directory to the same level as images directory
    """
    root = Path(r'G:\data\fepvd\v008\reolink\test\20240511')
    label_dirs = sorted(root.glob('**/predict/**/labels_xml'))
    for label_dir in tqdm(label_dirs):
        shutil.move(label_dir, label_dir.parents[2])


def cp_pred_into_video_dir(video_dir: [str, Path], exp: str,):
    """
    Copy predicted images and labels into the video directory
    """
    video_dir = Path(video_dir)
    cp(video_dir / 'predict' / exp, video_dir / 'images_vis_labels', '*.jpg')
    cp(video_dir / 'predict' / exp / 'labels', video_dir / 'labels', '*.txt')


def cp_pred_into_video_dirs():
    exp = 'wd_v008_004_epochs_300'
    root = Path(r'U:\Animal\Private\reolink\user_feedback\20240613')
    video_paths = sorted(root.glob('**/*.[am][opv][4iv]'))
    video_dirs = [p.parent / p.stem for p in video_paths]

    for i, video_dir in enumerate(video_dirs):
        print(f'[{i + 1} / {len(video_dirs)}] {video_dir}')
        cp_pred_into_video_dir(video_dir, exp)


def rm_pred_dirs():
    root = Path(r'G:\data\fepvd\v008\youtube')
    pred_dirs = sorted(root.glob('*/images_vis_labels'))
    for pred_dir in tqdm(pred_dirs):
        shutil.rmtree(pred_dir)


def main():
    # root = Path('/home/kemove/8TSSD/ganhao/data/wd/v008/labels_iqa/reolink/user')
    # video_dirs = sorted(p for p in root.glob('202[23]/*') if p.is_dir())
    # for i, video_dir in enumerate(video_dirs):
    #     print(f'[{i + 1} / {len(video_dirs)}] {video_dir}')
    #     mv_predicted_imgs_in_a_dir(video_dir)
    # save_fp_and_fn(
    #     r'Z:\8TSSD\ganhao\data\fepvd\v007\val.txt',
    #     r'Z:\8TSSD\ganhao\projects\ultralytics\runs\detect\fepvd\predict\fepvd_v007_001_mosaic_1_epochs_300_val\labels',
    #     r'G:\data\fepvd\vis_007_001_val',
    # )
    rm_pred_dirs()


if __name__ == '__main__':
    main()
