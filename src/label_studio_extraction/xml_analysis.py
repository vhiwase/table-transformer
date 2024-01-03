# import xml.etree.ElementTree as ET
# import pathlib
# import os

# def get_class_map(data_type):
#     if data_type == 'structure':
#         class_map = {
#             'table': 0,
#             'table column': 1,
#             'table row': 2,
#             'table column header': 3,
#             'table projected row header': 4,
#             'table spanning cell': 5,
#             'no object': 6
#         }
#     else:
#         class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
#     return class_map

# def get_unique_object_keys(xml_file):
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
#     names = []
#     poses = []
#     truncateds = []
#     difficults = []
#     occludeds = []
#     for object_ in root.iter('object'):
#         name = object_.find("name").text
#         names.append(name)
#         pose = object_.find("pose").text
#         poses.append(pose)
#         truncated = object_.find("truncated").text
#         truncateds.append(truncated)
#         difficult = object_.find("difficult").text
#         difficults.append(difficult)
#         occluded = object_.find("occluded").text
#         occludeds.append(occluded)
#         bboxes = []
#         for bndbox in object_.findall("bndbox"):
#             ymin = float(bndbox.find("ymin").text)
#             xmin = float(bndbox.find("xmin").text)
#             ymax = float(bndbox.find("ymax").text)
#             xmax = float(bndbox.find("xmax").text)
#             bbox = [xmin, ymin, xmax, ymax] # PASCAL VOC
#             bboxes.append(bbox)
#     return names, poses, truncateds, difficults, occludeds

# data_type = 'detection'
# class_map = get_class_map(data_type)
# name_set = set()
# pose_set = set()
# truncated_set = set()
# difficult_set = set()
# occluded_set = set()

# detection_xml_folder = 'C:/Users/VaibhavHiwase/OneDrive - TechnoMile/Documents/Python Scripts/ICI/Table/pubtables-1m/PubTables-1M-Detection/test'

# detection_xml_folder = pathlib.Path(detection_xml_folder)
# total_length = len(os.listdir(detection_xml_folder))
# for enum, xlm_name in  enumerate(os.listdir(detection_xml_folder)):
#     print(f'{total_length - enum} is remaining ...')
#     xml_file = detection_xml_folder / xlm_name
#     names, poses, truncateds, difficults, occludeds = get_unique_object_keys(xml_file)
#     for name in names:
#         name_set.add(name)
#     for pose in poses:
#         pose_set.add(pose)
#     for truncated in truncateds:
#         truncated_set.add(truncated)
#     for difficult in difficults:
#         difficult_set.add(difficult)
#     for occluded in occludeds:
#         occluded_set.add(occluded)


# detection_xml_folder = 'C:/Users/VaibhavHiwase/OneDrive - TechnoMile/Documents/Python Scripts/ICI/Table/pubtables-1m/PubTables-1M-Detection/val'

# detection_xml_folder = pathlib.Path(detection_xml_folder)
# total_length = len(os.listdir(detection_xml_folder))
# for enum, xlm_name in  enumerate(os.listdir(detection_xml_folder)):
#     print(f'{total_length - enum} is remaining ...')
#     xml_file = detection_xml_folder / xlm_name
#     names, poses, truncateds, difficults, occludeds = get_unique_object_keys(xml_file)
#     for name in names:
#         name_set.add(name)
#     for pose in poses:
#         pose_set.add(pose)
#     for truncated in truncateds:
#         truncated_set.add(truncated)
#     for difficult in difficults:
#         difficult_set.add(difficult)
#     for occluded in occludeds:
#         occluded_set.add(occluded)
    
#################################################################################################################

import xml.etree.ElementTree as ET
import pathlib
import os
from collections import defaultdict
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random

try:
    FILE_PATH = pathlib.Path(__file__)
except NameError:
    FILE_PATH = pathlib.Path('.xml_analysis.py')

ROOT_PATH = FILE_PATH.parent

def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6
        }
    else:
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map

def get_unique_object_keys(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    names = []
    poses = []
    truncateds = []
    difficults = []
    occludeds = []
    name_bbox_mapping = defaultdict(list)
    for object_ in root.iter('object'):
        name = object_.find("name").text
        names.append(name)
        pose = object_.find("pose").text
        poses.append(pose)
        truncated = object_.find("truncated").text
        truncateds.append(truncated)
        difficult = object_.find("difficult").text
        difficults.append(difficult)
        occluded = object_.find("occluded").text
        occludeds.append(occluded)
        bboxes = []
        for bndbox in object_.findall("bndbox"):
            ymin = float(bndbox.find("ymin").text)
            xmin = float(bndbox.find("xmin").text)
            ymax = float(bndbox.find("ymax").text)
            xmax = float(bndbox.find("xmax").text)
            bbox = [xmin, ymin, xmax, ymax] # PASCAL VOC
            bboxes.append(bbox)
        name_bbox_mapping[name].append(bboxes)
    return names, poses, truncateds, difficults, occludeds, name_bbox_mapping

def generate_unique_class_images(xml_folder_path, k=5):
    if 'PubTables-1M-Structure' in xml_folder_path:
        data_type = 'structure'
        table_label_images_path = ROOT_PATH / 'structure_table_label_images'
    else:
        data_type = 'detection'
        table_label_images_path = ROOT_PATH / 'detection_table_label_images'
    
    class_map = get_class_map(data_type)
    os.makedirs(table_label_images_path, exist_ok=True)
    for folder in class_map:
        os.makedirs(table_label_images_path / folder, exist_ok=True)
    
    xml_folder_path = pathlib.Path(xml_folder_path)
    xml_file_names = os.listdir(xml_folder_path)
    ##########################################
    xml_file_names = random.sample(xml_file_names, k)
    ##########################################
    total_length = len(xml_file_names)
    for enum, xlm_name in  enumerate(xml_file_names):
        print(f'{total_length - enum} is remaining ...')
        xml_file = xml_folder_path / xlm_name
        if isinstance(xml_file, pathlib.Path):
            xml_file = xml_file.absolute().as_posix()
        names, poses, truncateds, difficults, occludeds, name_bbox_mapping = get_unique_object_keys(xml_file)
        image_filepath = xml_file.replace('/val/', '/images/').replace('.xml', '.jpg')
        name_image_dict = {}
        for name in name_bbox_mapping:
            image = Image.open(image_filepath)
            image_name = pathlib.Path(image_filepath).name
            draw = ImageDraw.Draw(image)
            for table_spanning_cell in name_bbox_mapping[name]:
                for bbox in table_spanning_cell:
                    draw.rectangle(bbox,  outline='red')
            name_image_dict[name] = image
        for folder, image in name_image_dict.items():
            folder_path = table_label_images_path / folder
            image_path = folder_path / image_name
            image_path = image_path.absolute().as_posix()
            image.save(image_path)

if __name__ == '__main__':
    xml_folder_path = 'C:/Users/VaibhavHiwase/OneDrive - TechnoMile/Documents/Python Scripts/ICI/Table/pubtables-1m/PubTables-1M-Structure/val'
    generate_unique_class_images(xml_folder_path, k=100)
    xml_folder_path = 'C:/Users/VaibhavHiwase/OneDrive - TechnoMile/Documents/Python Scripts/ICI/Table/pubtables-1m/PubTables-1M-Detection/val'
    generate_unique_class_images(xml_folder_path, k=100)
    
            