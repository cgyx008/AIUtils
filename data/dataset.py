import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    import sys
    sys.path.insert(0, Path(__file__).parents[1].as_posix())
from file.ops import create_parent_dirs


def change_path(path, src_dir='images', dst_dir='labels', dst_suf='.txt'):
    parts = Path(path).parts
    new_path = Path(*(dst_dir if p == src_dir else p for p in parts))
    new_path = new_path.with_suffix(dst_suf)
    return new_path


def get_img_txt_xml(path):
    path = Path(path)
    parts = path.parts
    if 'images' in parts:
        img_path = path
        txt_path = change_path(path, 'images', 'labels', '.txt')
        xml_path = change_path(path, 'images', 'labels_xml', '.xml')

    elif 'labels' in parts:
        img_path = change_path(path, 'labels', 'images', '.jpg')
        txt_path = path
        xml_path = change_path(path, 'labels', 'labels_xml', '.xml')

    elif 'labels_xml' in parts:
        img_path = change_path(path, 'labels_xml', 'images', '.jpg')
        txt_path = change_path(path, 'labels_xml', 'labels', '.txt')
        xml_path = path
    else:
        raise ValueError('path should contain images or labels or labels_xml dir')

    return [img_path, txt_path, xml_path]


def split_train_val(root, make_copy=False):
    root = Path(root)
    img_paths = sorted(root.glob('images/**/*.[jp][pn]g'))
    img_names = {p.name for p in img_paths}
    assert len(img_paths) == len(img_names)
    random.shuffle(img_paths)

    num_val = len(img_paths) // 10
    train_imgs = sorted(img_paths[:-num_val])
    val_imgs = sorted(img_paths[-num_val:])

    train_lines = [f'{p.as_posix()}\n' for p in train_imgs]
    val_lines = [f'{p.as_posix()}\n' for p in val_imgs]
    with open(root / 'train.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(root / 'val.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    if make_copy:
        train_dir = root / 'train'
        val_dir = root / 'val'
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        val_imgs = {str(p) for p in val_imgs}
        for img_path in tqdm(img_paths):
            itx = get_img_txt_xml(img_path)

            dst_dir = val_dir if str(img_path) in val_imgs else train_dir

            dst_img = dst_dir / img_path.relative_to(root)
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img)

            if itx[1].exists():
                dst_txt = dst_dir / itx[1].relative_to(root)
                dst_txt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(itx[1], dst_txt)

            if itx[2].exists():
                dst_xml = dst_dir / itx[2].relative_to(root)
                dst_xml.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(itx[2], dst_xml)


def rm_dup_imgs(txt_path):
    # txt_path = Path('/home/kemove/8TSSD/ganhao/data/wd/v04/trainval.txt')
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    save_txt = txt_path.with_stem(f'{txt_path.stem}_ids')
    for line in tqdm(lines, ascii=True):
        img_path = line.strip()
        with Image.open(img_path) as img:
            w, h = img.size
            s = np.sum(img)
        with open(save_txt, 'a', encoding='utf-8') as f:
            f.write(f'{img_path} {w}_{h}_{s}\n')


def cp_dum_imgs(txt_path):
    # txt_path = Path('/home/kemove/8TSSD/ganhao/data/wd/v04/trainval_ids.txt')
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    id2imgs = {}
    for line in tqdm(lines, ascii=True):
        img, iden = line.rsplit(maxsplit=1)
        if iden not in id2imgs:
            id2imgs[iden] = [img]
        else:
            id2imgs[iden].append(img)

    dup_imgs = {k: v for k, v in id2imgs.items() if len(v) != 1}
    for k, v in tqdm(dup_imgs.items(), ascii=True):
        with Image.open(v[0]) as img:
            img = np.array(img)
        for p in v[1:]:
            with Image.open(p) as img1:
                img1 = np.array(img1)
            if not np.array_equal(img, img1):
                v.remove(p)
        dup_imgs[k] = v
    dup_imgs = {k: v for k, v in dup_imgs.items() if len(v) != 1}
    with open(txt_path.parent / 'dup_imgs.txt', 'w', encoding='utf-8') as f:
        for k, v in tqdm(dup_imgs.items()):
            f.write(k)
            for p in v:
                f.write(f' {p}')
            f.write('\n')
    with open(txt_path.parent / 'rm_dup_imgs.txt', 'w', encoding='utf-8') as f:
        for v in tqdm(dup_imgs.values()):
            for p in v[1:]:
                f.write(f'{p}\n')

    dup_dir = txt_path.parent / 'dup_imgs'
    dup_dir.mkdir(exist_ok=True)
    for k, v in tqdm(dup_imgs.items(), ascii=True):
        dst = dup_dir / k
        dst.mkdir(exist_ok=True)
        for p in v:
            shutil.copy2(p, dst)


def del_dup_imgs(root, txt):
    # root = Path('/home/kemove/8TSSD/ganhao/data/wd/v04/rm_dup_imgs.txt')
    with open(root, 'r', encoding='utf-8') as f:
        dup = set(f.readlines())

    # txt = Path('/home/kemove/8TSSD/ganhao/data/wd/v04/val.txt')
    with open(txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line for line in lines if line not in dup]
    with open(txt, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def split_tasks(root):
    """
    Split and assign clean tasks.
    """
    # root = Path(r'U:\Animal')
    img_paths = sorted(root.glob('**/images/*.jpg'))

    num_half = len(img_paths) // 2
    split_path = img_paths[num_half]
    split_dir_imgs = sorted(split_path.parent.glob('*.jpg'))
    num_contained_imgs = split_dir_imgs.index(split_path)
    num_split_dir_imgs = len(split_dir_imgs)
    if_contain = num_contained_imgs / num_split_dir_imgs > 0.5

    contain_dirs = sorted({str(p.parents[1]) for p in img_paths[:num_half]})
    remain_dirs = sorted({str(p.parents[1]) for p in img_paths[num_half:]})
    if if_contain:
        remain_dirs.remove(str(split_path.parents[1]))
    else:
        contain_dirs.remove(str(split_path.parents[1]))

    contain_dir = root / '0'
    contain_dir.mkdir(exist_ok=True)
    for d in tqdm(contain_dirs):
        shutil.copytree(d, contain_dir / Path(d).name)

    remain_dir = root / '1'
    remain_dir.mkdir(exist_ok=True)
    for d in tqdm(remain_dirs):
        shutil.copytree(d, remain_dir / Path(d).name)


def update_labels_from_a_dir(src_dir, dst_dir):
    # src_dir = Path(r'Z:\8TSSD\ganhao\data\fepvd\v006')
    # dst_dir = Path(r'T:\Private')

    src_paths = sorted(src_dir.glob('**/labels_xml/*.xml'))
    src_stem2path = {p.stem: p for p in src_paths}
    dst_paths = sorted(dst_dir.glob('2024*/**/labels_xml/*.xml'))

    missings = [p for p in dst_paths if p.stem not in src_stem2path]
    if missings:
        print(missings)
        raise FileNotFoundError('Files not found!')

    for dst in tqdm(dst_paths):
        src = src_stem2path[dst.stem]
        shutil.copy2(src, dst)


def cp_dataset(txt_path, dst_dir):
    """Copy data from txt to dst_dir
    Args:
        txt_path (str | Path): path to train.txt or val.txt
        dst_dir (str | Path): path to destination folder
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    src_img_paths = [Path(line.strip()) for line in tqdm(lines)]
    src_txt_paths = [get_img_txt_xml(p)[1] for p in tqdm(src_img_paths)]
    src_paths = src_img_paths + src_txt_paths

    dst_img_paths = [
        dst_dir / p.relative_to(Path(*p.parts[:p.parts.index('images')]))
        for p in tqdm(src_img_paths)
    ]
    dst_txt_paths = [get_img_txt_xml(p)[1] for p in tqdm(dst_img_paths)]
    dst_paths = dst_img_paths + dst_txt_paths

    # Create parent directories
    create_parent_dirs(dst_paths)

    for src, dst in zip(tqdm(src_paths), dst_paths):
        shutil.copy2(src, dst)


def vis_imgs_without_obj(root, num_threads=8):
    root = Path(root)

    vis_dir = root / 'images_without_object'
    vis_dir.mkdir(exist_ok=True)

    img_paths = sorted(root.glob('images/**/*.jpg'))

    def _vis_imgs_without_obj(img_path):
        txt_path = get_img_txt_xml(img_path)[1]
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            with open(vis_dir / 'img_paths.txt', 'a', encoding='utf-8') as f:
                f.write(f'{img_path}\n')
            shutil.copy2(img_path, vis_dir)

    with ThreadPoolExecutor(num_threads) as executor:
        list(tqdm(executor.map(_vis_imgs_without_obj, img_paths),
                  total=len(img_paths)))


def merge_txts(root):
    root = Path(root)
    for mode in ['train', 'val']:
        # txt_paths = [root / f'{mode}.txt']
        txt_paths = (sorted(root.glob(f'**/{mode}.txt')))

        lines = []
        for txt_path in tqdm(txt_paths):
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines.extend(f.readlines())

        with open(root / f'{mode}_new.txt', 'w', encoding='utf-8') as f:
            f.writelines(lines)


def rm_old_dirs(root):
    """Remove old directories like 'labels' and 'images_vis_labels'"""
    # root = Path(r'Z:\8TSSD\ganhao\data\fepvd\v007')
    rm_dirs = sorted(list(root.glob('*/frames'))
                     + list(root.glob('*/predict')))
    for rm_dir in tqdm(rm_dirs):
        shutil.rmtree(rm_dir)


def cp_labels(src_dir, dst_dir):
    """Use labels in src_dir to update labels in dst_dir"""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    xml_paths = sorted(src_dir.glob('labels_xml/*.xml'))
    (dst_dir / 'labels_xml').mkdir(exist_ok=True)
    for xml_path in xml_paths:
        shutil.copy2(xml_path, dst_dir / 'labels_xml')


def update_labels(src, dst):
    """Use labels in src to update labels in dst"""
    src, dst = Path(src), Path(dst)
    src_videos = sorted(src.glob('**/images'))
    src_videos = {p.parts[-2]: p.parent for p in tqdm(src_videos)}
    dst_videos = sorted(dst.glob('**/images'))
    dst_videos = {p.parts[-2]: p.parent for p in tqdm(dst_videos)}
    assert len(src_videos) == len(dst_videos)
    video2dirs = {k: [src_videos[k], dst_videos[k]] for k in src_videos}

    for k, v in tqdm(video2dirs.items()):
        src_dir, dst_dir = v
        cp_labels(src_dir, dst_dir)


def jpg2npy(root):
    # oiv7: 1.4TB
    # o365: 500GB
    # root = Path('/home/ganhao/data/Objects365_v1/2019-08-02')
    img_paths = sorted(root.glob('images/train/*.jpg'))
    npy_sizes = []
    for img_path in tqdm(img_paths):
        with Image.open(img_path) as im:
            w, h = im.size
        r = 640 / max(h, w)
        w = round(w * r)
        h = round(h * r)
        npy_sizes.append(w * h * 3)

    print(f'need storage: {sum(npy_sizes) / (1024 ** 3)}GB')


def copy_and_save_as_640_npy(root):
    # root = Path('/data_raid0/ganhao/data/wd/v009')
    train_txt = root / 'train.txt'
    val_txt = root / 'val.txt'
    test_txt = root / 'test.txt'
    with open(train_txt, 'r', encoding='utf-8') as f:
        img_paths = [Path(p.strip()) for p in f]
    with open(val_txt, 'r', encoding='utf-8') as f:
        img_paths.extend([Path(p.strip()) for p in f])
    with open(test_txt, 'r', encoding='utf-8') as f:
        img_paths.extend([Path(p.strip()) for p in f])

    # npy_dir = root / 'images'
    # os.umask(0)
    # npy_dir.mkdir(exist_ok=True)

    def _letterbox_and_save_as_640_npy(img_path):
        npy_path = img_path.parent / f'{img_path.stem}.npy'
        if npy_path.exists():
            return
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        rw, rh = 640 / w, 352 / h
        r = min(rw, rh)
        w = round(w * r)
        h = round(h * r)
        img = cv2.resize(img, (w, h))
        np.save(npy_path, img)

    with ThreadPoolExecutor(4) as executor:
        list(tqdm(executor.map(_letterbox_and_save_as_640_npy, img_paths),
                  total=len(img_paths)))

    # for img_path in tqdm(img_paths):
    #     img = cv2.imread(str(img_path))
    #     h, w = img.shape[:2]
    #     rw, rh = 640 / w, 352 / h
    #     r = min(rw, rh)
    #     w = round(w * r)
    #     h = round(h * r)
    #     img = cv2.resize(img, (w, h))
    #     npy_path = img_path.parent / f'{img_path.stem}.npy'
    #     if not npy_path.exists():
    #         np.save(npy_path, img)


def main():
    copy_and_save_as_640_npy(Path(''))


if __name__ == '__main__':
    main()
