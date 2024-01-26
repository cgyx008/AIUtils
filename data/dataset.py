import random
import shutil
from pathlib import Path

from tqdm import tqdm


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
        if not img_path.exists():
            img_path = img_path.with_suffix('.png')
        txt_path = path
        xml_path = change_path(path, 'labels', 'labels_xml', '.xml')

    elif 'labels_xml' in parts:
        img_path = change_path(path, 'labels_xml', 'images', '.jpg')
        if not img_path.exists():
            img_path = img_path.with_suffix('.png')
        txt_path = change_path(path, 'labels_xml', 'labels', '.txt')
        xml_path = path
    else:
        raise ValueError('path should contain images or labels or labels_xml dir')

    return [img_path, txt_path, xml_path]


def split_train_val(root, make_copy=False):
    root = Path(root)
    img_paths = sorted(root.glob('images/**/*.[jp][pn]g'))
    img_names = list({p.name for p in img_paths})
    assert len(img_paths) == len(img_names)
    random.shuffle(img_names)

    num_val = len(img_paths) // 10
    train_imgs = sorted(img_paths[:-num_val])
    val_imgs = sorted(img_paths[-num_val:])

    train_lines = [f'{n}\n' for n in train_imgs]
    val_lines = [f'{n}\n' for n in val_imgs]
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


def main():
    split_train_val(r'G:\data\wr', True)


if __name__ == '__main__':
    main()
