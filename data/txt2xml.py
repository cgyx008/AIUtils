import os
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

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


def xml2txt(xml_path, txt_path=None, classes=('animal', 'person', 'vehicle')):
    # Initialize `txt_path` if not specified
    if txt_path is None:
        txt_path = Path(xml_path).with_suffix('.txt')
    # Initialize classes
    classes = {c: i for i, c in enumerate(classes)}

    # Get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image width and height
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    # Get object
    objs = root.findall('object')
    txt_lines = []
    for obj in objs:
        # Get class_id
        name = obj.find('name').text.lower()
        class_id = classes.get(name, -1)
        if class_id == -1:
            print(f'Error class name "{name}" in {xml_path}, skip it.')
            continue
        # Get box
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        w, h = xmax - xmin, ymax - ymin
        xc, yc = xmin + w/2, ymin + h/2
        # Normalize
        xc, yc, w, h = xc / width, yc / height, w / width, h / height
        txt_lines.append(f'{class_id} {xc} {yc} {w} {h}\n')

    # Write in txt
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(txt_lines)


def write_xml(img_path, xml_path):
    img_path, xml_path = Path(img_path), Path(xml_path)
    # Get width and height
    img = cv2.imread(str(img_path))
    if img is None:
        with Image.open(img_path) as img:
            img = np.ascontiguousarray(np.array(img)[..., ::-1])
    h, w = img.shape[:2]

    # Create new xml
    root = ET.Element('annotation')
    # folder
    folder = ET.SubElement(root, 'folder')
    folder.text = img_path.parts[-2]
    # filename
    filename = ET.SubElement(root, 'filename')
    filename.text = img_path.name
    # path
    path = ET.SubElement(root, 'path')
    path.text = str(img_path)
    # source
    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    # size
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    # segmented
    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'
    # object

    # Convert ElementTree to string
    xml_string = ET.tostring(root, encoding='utf-8')
    xml_string = ''.join(xml_string.decode('utf-8').split())

    # Parse the XML string using minidom
    dom = minidom.parseString(xml_string)
    pretty_xml_string = dom.toprettyxml()

    # Write the prettified XML string to a file
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_string)


def main():
    txt2xml(r'W:\ganhao\AD\wd\v04')


if __name__ == '__main__':
    main()
