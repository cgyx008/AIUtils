import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    import sys
    sys.path.append(Path(__file__).parents[1].as_posix())
from data.dataset import get_img_txt_xml
from file.ops import create_parent_dirs

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


def read_txt(txt_path, w=1, h=1):
    label = {'path': str(txt_path), 'width': w, 'height': h}

    # Read txt
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [list(map(eval, line.split())) for line in lines]

    obj_list = []
    for line in lines:
        class_id = line[0]

        # Normalized xcycwh
        nxc, nyc, nw, nh = line[1:5]
        # Normalized x1y1x2y2
        nxmin = nxc - nw/2
        nymin = nyc - nh/2
        nxmax = nxc + nw/2
        nymax = nyc + nh/2
        # Pixel xcycwh
        pxmin = int(nxmin * w)
        pymin = int(nymin * h)
        pxmax = int(nxmax * w)
        pymax = int(nymax * h)
        # Pixel x1y1x2y2
        pxc = int(nxc * w)
        pyc = int(nyc * h)
        pw = int(nw * w)
        ph = int(nh * h)

        conf, iqa = 0, 0
        if len(line) == 6:
            conf = line[5]
        if len(line) == 7:
            conf, iqa = line[5], line[6]

        obj_list.append({'class_id': class_id,
                         'pxyxy': [pxmin, pymin, pxmax, pymax],
                         'pxywh': [pxc, pyc, pw, ph],
                         'nxyxy': [nxmin, nymin, nxmax, nymax],
                         'nxywh': [nxc, nyc, nw, nh],
                         'conf': conf,
                         'iqa': iqa,})

    label['objects'] = obj_list
    return label


def txt2xml(root, classes=('animal', 'person', 'vehicle')):
    """Transform txts in the root into xmls"""
    # Glob txts
    txt_paths = sorted(Path(root).glob('**/labels/**/*.txt'))
    for txt_path in tqdm(txt_paths):
        if txt_path.stem == 'classes':
            continue
        # Get xml path
        img_path, _, xml_path = get_img_txt_xml(txt_path)
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

    label = read_xml(xml_path)
    txt_lines = []
    if not label['has_difficult']:
        for obj in label['objects']:
            # Get class_id
            name = obj['name']
            class_id = classes.get(name, -1)
            if class_id == -1:
                print(f'Error class name "{name}" in {xml_path}, skip it.')
                continue

            nxc, nyc, nw, nh = obj['nxywh']
            txt_lines.append(f'{class_id} {nxc} {nyc} {nw} {nh}\n')

    # Write in txt
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(txt_lines)


def read_xml(xml_path):
    label = {'path': str(xml_path)}
    # Get root
    tree = ET.parse(xml_path)
    root = tree.getroot()
    label['root'] = root

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
        # Pixel x1y1x2y2
        pxmin = int(obj.find('bndbox').find('xmin').text)
        pymin = int(obj.find('bndbox').find('ymin').text)
        pxmax = int(obj.find('bndbox').find('xmax').text)
        pymax = int(obj.find('bndbox').find('ymax').text)
        # Pixel xcycwh
        pxc = (pxmin + pxmax) // 2
        pyc = (pymin + pymax) // 2
        pw = pxmax - pxmin
        ph = pymax - pymin
        # Normalized x1y1x2y2
        nxmin = pxmin / width
        nymin = pymin / height
        nxmax = pxmax / width
        nymax = pymax / height
        # Normalized xcycwh
        nxc = pxc / width
        nyc = pyc / height
        nw = pw / width
        nh = ph / height

        # Get difficult
        difficult = int(obj.find('difficult').text)
        has_difficult = has_difficult or bool(difficult)

        # Append in obj_list
        obj_list.append({'name': name,
                         'pxyxy': [pxmin, pymin, pxmax, pymax],
                         'pxywh': [pxc, pyc, pw, ph],
                         'nxyxy': [nxmin, nymin, nxmax, nymax],
                         'nxywh': [nxc, nyc, nw, nh],
                         'difficult': difficult})

    label['has_difficult'] = has_difficult
    label['objects'] = obj_list
    for class_name in class_names:
        label[class_name] = [
            {'pxyxy': obj['pxyxy'], 'difficult': obj['difficult']}
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
        # Read xml
        label = read_xml(xml_path)
        root = label['root']
        img_area = label['width'] * label['height']

        num_objs += len(label['objects'])
        for obj in label['objects']:
            obj_area = obj['pxywh'][2] * obj['pxywh'][3]  # noqa
            if obj_area / img_area < 10**2 / 512**2:  # noqa
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
    create_parent_dirs(txt_paths)

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


def write_down_difficult_imgs():
    root = Path(r'G:\data\wd\working\clean_train_val_v05_002')
    xmls = sorted(root.glob('labels_xml/*.xml'))
    difficult_xmls = [p for p in tqdm(xmls) if read_xml(p)['has_difficult']]
    with open(root / 'difficult_xmls.txt', 'w', encoding='utf-8') as f:
        for difficult_xml in difficult_xmls:
            f.write(f'{difficult_xml}\n')


def cnt_labels(xml_dir, class_names=('animal', 'person', 'vehicle')):
    name2num = dict.fromkeys(class_names, 0)
    xml_paths = sorted(Path(xml_dir).glob('**/*.xml'))
    for xml_path in tqdm(xml_paths):
        label = read_xml(xml_path)
        for c in class_names:
            name2num[c] += len(label.get(c, []))
    print(name2num)


def rm_extra_txts(root):
    root = Path(root)
    txt_paths = sorted(root.glob('labels_iqa/**/*.txt'))
    rm_paths = [p
                for p in tqdm(txt_paths) if not get_img_txt_xml(p)[0].exists()]
    for p in tqdm(rm_paths):
        p.unlink()


def rm_extra_xmls(root):
    root = Path(root)

    img_paths = sorted(root.glob('images/**/*.jpg'))
    img_stems = {p.stem for p in tqdm(img_paths)}

    xml_paths = sorted(root.glob('labels_xml/**/*.xml'))
    xml_stems = {p.stem for p in tqdm(xml_paths)}

    extra_xml_stems = xml_stems - img_stems
    extra_xml_paths = [p for p in tqdm(xml_paths) if p.stem in extra_xml_stems]
    for p in tqdm(extra_xml_paths):
        p.unlink()


def crop_objs(img_dir, txt_dir):
    img_dir, txt_dir = Path(img_dir), Path(txt_dir)
    img_paths = sorted(img_dir.glob('**/*.jpg'))
    txt_paths = sorted(txt_dir.glob('**/*.txt'))

    name_fmt = '{}_xyxy_{}_{}_{}_{}.jpg'

    stem2txt_path = {p.stem: p for p in tqdm(txt_paths)}
    for img_path in tqdm(img_paths):
        if img_path.stem not in stem2txt_path:
            continue
        txt_path = stem2txt_path[img_path.stem]

        img = cv2.imread(str(img_path))
        label = read_txt(txt_path, img.shape[1], img.shape[0])

        for obj in label['objects']:
            box = obj['pxyxy']
            img_obj = img[box[1]:box[3], box[0]:box[2]]

            name = name_fmt.format(img_path.stem, *box)
            save_path = img_path.parents[1] / 'objects' / name
            save_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(save_path), img_obj)


def append_0_in_txt(txt_path, save_path=None):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() + ' 0.0\n' for line in lines]

    if save_path is None:
        txt_path = Path(txt_path)
        save_path = txt_path.with_stem(f'{txt_path.stem}_append_0.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def append_0_in_labels_iqa_dir(root):
    txt_paths = sorted(root.glob('labels/*.txt'))
    save_dir = root / 'labels_iqa'
    save_dir.mkdir(parents=True, exist_ok=True)
    for txt_path in tqdm(txt_paths):
        append_0_in_txt(txt_path, save_dir / txt_path.name)


def main():
    root = Path(r'G:\data\fepvd\v008\reolink\test')
    video_dirs = sorted(p for p in root.glob('2024051*/*') if p.is_dir())
    for i, video_dir in enumerate(video_dirs):
        print(f'[{i + 1} / {len(video_dirs)}] {video_dir}')
        create_empty_labels(video_dir)
        # xmls2txts(video_dir, ('person', 'vehicle'))
        # append_0_in_labels_iqa_dir(video_dir)
        # txt2xml(video_dir, ('person', 'vehicle'))
        # create_empty_labels(video_dir)


if __name__ == '__main__':
    main()
