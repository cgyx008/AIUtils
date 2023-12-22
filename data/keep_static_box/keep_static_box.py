import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.dom import minidom

import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
from tqdm import tqdm


def get_xml_labels(xml_path):
    # Get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get object
    cat2boxes = defaultdict(list)
    objs = root.findall('object')
    for obj in objs:
        # Get class_id
        class_name = obj.find('name').text.lower()
        # Get box
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        cat2boxes[class_name].append([xmin, ymin, xmax, ymax])

    return cat2boxes, root


def write_xml(xml_path, root_orig, labels):
    xml_path = Path(xml_path)

    # Create new xml
    root = ET.Element('annotation')
    # folder
    folder = ET.SubElement(root, 'folder')
    folder.text = root_orig.find('folder').text
    # filename
    filename = ET.SubElement(root, 'filename')
    filename.text = root_orig.find('filename').text
    # path
    path = ET.SubElement(root, 'path')
    path.text = root_orig.find('path').text
    # source
    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    # size
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = root_orig.find('size').find('width').text
    height = ET.SubElement(size, 'height')
    height.text = root_orig.find('size').find('height').text
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    # segmented
    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'

    for cat_name, boxes in labels.items():
        for box in boxes:
            obj = ET.SubElement(root, 'object')
            name = ET.SubElement(obj, 'name')
            name.text = cat_name  # name
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(box[0])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(box[1])
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(box[2])
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(box[3])

    # Convert ElementTree to string
    xml_string = ET.tostring(root, encoding='utf-8')
    xml_string = ''.join(xml_string.decode('utf-8').split())

    # Parse the XML string using minidom
    dom = minidom.parseString(xml_string)
    pretty_xml_string = dom.toprettyxml()

    # Write the prettified XML string to a file
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_string)


def keep_static_box(iou_threshold=0.6):
    # 获取当前时间
    current_time = datetime.now()

    # 将时间格式化为指定的字符串格式
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')

    standard_dir = Path('./standard_xml')
    replace_dir = Path('./replace_xml')
    backup_dir = Path(f'./backup_xml/{formatted_time}')

    standard_xml = sorted(standard_dir.glob('*.xml'))[0]
    replace_xmls = sorted(replace_dir.glob('*.xml'))
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 备份
    print(f'正在将standard_xml文件夹和replace_xml文件夹'
          f'备份到backup_xml/{formatted_time}文件夹中...')
    backup_standard_dir = backup_dir / 'stardard_xml'
    backup_replace_dir = backup_dir / 'replace_xml'
    backup_standard_dir.mkdir(parents=True, exist_ok=True)
    backup_replace_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(standard_xml, backup_standard_dir)
    for xml in tqdm(replace_xmls):
        shutil.copy2(xml, backup_replace_dir)
    print('\n完成备份！')

    # 读取标准框
    labels1, _ = get_xml_labels(standard_xml)

    # 比较替换框
    print('正在将标准框应用于其他标签...')
    for xml in tqdm(replace_xmls):
        labels2, root = get_xml_labels(xml)
        for k, boxes1 in labels1.items():
            boxes2 = labels2.get(k, None)
            if boxes2 is None:
                labels2[k] = boxes1
                continue

            iou = box_iou(torch.tensor(boxes1), torch.tensor(boxes2))
            iou = iou * (iou > iou_threshold)
            iou = iou.numpy()

            boxes1_ids, boxes2_ids = linear_sum_assignment(iou, maximize=True)
            for boxes1_id, boxes2_id in zip(boxes1_ids, boxes2_ids):
                if iou[boxes1_id, boxes2_id] > 0:
                    boxes2[boxes2_id] = boxes1[boxes1_id]
                else:
                    boxes2.append(boxes1[boxes1_id])

            # 添加漏检目标
            fns = [boxes1[i] for i in range(len(boxes1)) if i not in boxes1_ids]
            boxes2.extend(fns)
        write_xml(xml, root, labels2)
    print('完成应用！')


def main():
    keep_static_box()


if __name__ == '__main__':
    main()
