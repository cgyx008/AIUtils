import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from data.dataset import get_img_txt_xml

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


def txt2xml(root, classes=('animal', 'person', 'vehicle')):
    """Transform txts in the root into xmls"""
    # Glob txts
    txt_paths = sorted(Path(root).glob('labels/*.txt'))
    s = os.sep  # '/' in Linux, '\\' in Windows
    for txt_path in tqdm(txt_paths):
        if txt_path.stem == 'classes':
            continue
        # Get xml path
        xml_path = txt_path.with_suffix('.xml')
        xml_path = str(xml_path).replace(f'{s}labels{s}', f'{s}labels_xml{s}')
        xml_path = Path(xml_path)
        if xml_path.exists():
            continue

        # Read txt
        with open(txt_path, 'r', encoding='utf-8') as f:
            labels = np.array([list(map(eval, line.split())) for line in f])
        if labels.size == 0:
            labels = np.zeros((1, 5)) - 1  # Compatibility with empty images
        class_ids, boxes = labels[:, 0].astype(int), labels[:, 1:]

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
        if not img_path.exists():
            continue
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
            for class_id, box in zip(class_ids, boxes):
                if class_id == -1:
                    continue
                c = classes[class_id]
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
        if obj.find('difficult').text == '1':
            txt_lines.clear()
            break
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


def read_xml(xml_path):
    label = {'path': str(xml_path)}
    # Get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image width and height
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    label['width'] = width
    label['height'] = height

    # Get object
    objs = root.findall('object')
    has_difficult = False
    class_names = set()
    obj_list = []
    for obj in objs:
        # Get class_id
        name = obj.find('name').text.lower()
        class_names.add(name)
        # Get box
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        # Get difficult
        difficult = int(obj.find('difficult').text)
        has_difficult = has_difficult or bool(difficult)

        obj_list.append({'name': name,
                         'box': [xmin, ymin, xmax, ymax],
                         'difficult': difficult})
    label['has_difficult'] = has_difficult
    label['objects'] = obj_list
    for class_name in class_names:
        label[class_name] = [
            {'box': obj['box'], 'difficult': obj['difficult']}
            for obj in obj_list if obj['name'] == class_name
        ]

    return label


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
    # obj = ET.SubElement(root, 'object')
    # name = ET.SubElement(obj, 'name')
    # name.text = 'Vehicle'  # name
    # pose = ET.SubElement(obj, 'pose')
    # pose.text = 'Unspecified'
    # truncated = ET.SubElement(obj, 'truncated')
    # truncated.text = '0'
    # difficult = ET.SubElement(obj, 'difficult')
    # difficult.text = '0'
    # bndbox = ET.SubElement(obj, 'bndbox')
    # xmin = ET.SubElement(bndbox, 'xmin')
    # xmin.text = '0'
    # ymin = ET.SubElement(bndbox, 'ymin')
    # ymin.text = '0'
    # xmax = ET.SubElement(bndbox, 'xmax')
    # xmax.text = '0'
    # ymax = ET.SubElement(bndbox, 'ymax')
    # ymax.text = '0'

    # Convert ElementTree to string
    xml_string = ET.tostring(root, encoding='utf-8')
    xml_string = ''.join(xml_string.decode('utf-8').split())

    # Parse the XML string using minidom
    dom = minidom.parseString(xml_string)
    pretty_xml_string = dom.toprettyxml()

    # Write the prettified XML string to a file
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_string)


def remove_small_objs():
    cwd = Path(r'T:\Working\v03\train')
    xml_paths = sorted(cwd.glob('labels_xml/*.xml'))
    num_objs, num_small_objs = 0, 0
    has_small_obj_xml = set()
    for xml_path in tqdm(xml_paths):
        # Get root
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Size
        w = int(root.find('size').find('width').text)
        h = int(root.find('size').find('height').text)
        img_area = w * h

        # Get object
        objs = root.findall('object')
        num_objs += len(objs)
        for obj in objs:
            # Get class_id
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            w, h = xmax - xmin, ymax - ymin
            area = w * h
            if area / img_area < 10**2 / 512**2:
                num_small_objs += 1
                has_small_obj_xml.add(xml_path)
                root.remove(obj)

        # Convert ElementTree to string
        xml_string = ET.tostring(root, encoding='utf-8')
        xml_string = ''.join(xml_string.decode('utf-8').split())
        xml_string = xml_string.replace('<annotationverified="yes">',
                                        '<annotation verified="yes">')

        # Parse the XML string using minidom
        dom = minidom.parseString(xml_string)
        pretty_xml_string = dom.toprettyxml()

        # Write the prettified XML string to a file
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml_string)
    print(num_objs, num_small_objs)


def xmls2txts(root, classes=('animal', 'person', 'vehicle')):
    # xml2txt
    root = Path(root)

    # xml
    xml_dir = root / 'labels_xml'
    xml_paths = sorted(xml_dir.glob('**/*.xml'))

    # txt
    txt_dir = root / 'labels'
    txt_dir.mkdir(exist_ok=True)
    txt_paths = [(txt_dir / xml_path.relative_to(xml_dir)).with_suffix('.txt')
                 for xml_path in tqdm(xml_paths)]
    txt_parents = {str(txt_path.parent) for txt_path in tqdm(txt_paths)}
    for txt_parent in tqdm(txt_parents):
        Path(txt_parent).mkdir(parents=True, exist_ok=True)

    # classes
    classes = [classes] * len(xml_paths)

    with ThreadPoolExecutor(8) as executor:
        list(tqdm(executor.map(xml2txt, xml_paths, txt_paths, classes),
                  total=len(xml_paths)))


def create_empty_labels(root):
    root = Path(root)
    img_paths = sorted(root.glob('images/**/*.[jp][pn]g'))
    for img_path in tqdm(img_paths):
        txt_path, xml_path = get_img_txt_xml(img_path)[1:]
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.parent.mkdir(parents=True, exist_ok=True)

        if not txt_path.exists():
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('')
        if not xml_path.exists():
            write_xml(img_path, xml_path)


def main():
    create_empty_labels(
        r'G:\data\wd\working\v05\reolink\user_feedback\20240222_background_fp')


if __name__ == '__main__':
    main()
