import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


r"""VOC format:
<annotation>
    <folder>images</folder>
    <filename>5858bf4a-23d2-11e8-a6a3-ec086b02610b.jpg</filename>
    <path>W:\ganhao\AD\caltech\0003_raccoon\images\5858bf4a-23d2-11e8-a6a3-ec086b02610b.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>2048</width>
        <height>1494</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>Animal</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>1</difficult>
        <bndbox>
            <xmin>1099</xmin>
            <ymin>698</ymin>
            <xmax>1305</xmax>
            <ymax>1019</ymax>
        </bndbox>
    </object>
</annotation>
"""

start_fmt = r'''<annotation>
    <folder>images</folder>
    <filename>{}</filename>
    <path>{}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
'''
obj_fmt = r'''    <object>
        <name>{}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
'''
end_fmt = '''</annotation>
'''


def txt2xml(root):
    """Transform txts in the root into xmls"""
    # Glob txts
    txt_paths = sorted(list(Path(root).glob('**/*.txt')))
    s = os.sep  # '/' in Linux, '\\' in Windows
    for txt_path in tqdm(txt_paths):
        # Get xml path
        xml_path = txt_path.with_suffix('.xml')
        xml_path = str(xml_path).replace(f'{s}labels{s}', f'{s}labels_voc{s}')
        xml_path = Path(xml_path)
        if xml_path.exists():
            continue

        # Read txt
        with open(txt_path, 'r', encoding='utf-8') as f:
            labels = np.array([list(map(eval, line.split())) for line in f])
        if labels.size == 0:
            labels = np.zeros((1, 5)) - 1  # Compatibility with empty images
        classes, boxes = labels[:, 0].astype(int), labels[:, 1:]

        # (xc, yc, w, h) norm -> (xmin, ymin, xmax, ymax) norm
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        # Read the image
        img_path = str(txt_path).replace(f'{s}labels{s}', f'{s}images{s}')
        img_path = Path(img_path).with_suffix('.jpg')
        if not img_path.exists():
            img_path = img_path.with_suffix('.png')
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
        else:
            with Image.open(img_path) as img:
                w, h = img.size

        # norm -> pixel
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = boxes.astype(int)

        # Write in xml
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(start_fmt.format(img_path.name, str(img_path), w, h))
            for class_id, box in zip(classes, boxes):
                if class_id == -1:
                    continue
                c = ['Animal', 'Person', 'Vehicle'][class_id]
                f.write(obj_fmt.format(c, *box))
            f.write(end_fmt)


def main():
    txt2xml(r'W:\ganhao\AD\wd\v04')


if __name__ == '__main__':
    main()
