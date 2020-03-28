import os
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd


annotations_dir = 'VOCdevkit/VOC2012/Annotations'
imgs_dir = 'VOCdevkit/VOC2012/JPEGImages'
preprocessed_dir = 'VOCdevkit/VOC2012'


def xml_parse(tree):
    obj = 0
    img_rows = OrderedDict()

    for attrs in tree.iter():

        if attrs.tag == 'filename':
            img_rows[attrs.tag] = str(attrs.text)

        if attrs.tag == 'size':
            for attr in attrs:
                img_rows[attr.tag] = int(attr.text)

        if attrs.tag == 'object':
            for attr in attrs:
                if attr.tag == 'name':
                    img_rows['bbox_{}_{}'.format(obj, attr.tag)] = str(attr.text)
                if attr.tag == 'bndbox':
                    for box in attr:
                        img_rows['bbox_{}_{}'.format(obj, box.tag)] = float(box.text)
                    obj += 1
    img_rows['obj'] = obj

    return img_rows



annotations_res_csv = []
test = 0
for xml_file in os.listdir(annotations_dir):
    tree = ET.parse(os.path.join(annotations_dir, xml_file))
    element = xml_parse(tree)
    annotations_res_csv.append(element)
    test += 1
    if test == 10:
        break

annotations_res_csv = pd.DataFrame(annotations_res_csv)
# annotations_res_csv.to_csv(os.path.join(preprocessed_dir, 'annotations.csv'), index=False)
print(annotations_res_csv.head())