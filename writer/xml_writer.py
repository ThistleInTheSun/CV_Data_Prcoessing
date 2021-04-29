#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/21
# @Author  : xq


from xml.etree import ElementTree as ET
from lxml import etree
import os


class XmlWriter(object):
    def __init__(self, path):
        self.save_path = path

    def gen_xml(self, boxes, image, name, ratio=1):
        prefix = name.split('.jpg')[0]
        root = etree.Element("annotation")
        file_name = etree.SubElement(root, "filename")
        file_name.text = prefix
        url = etree.SubElement(root, "url")
        url.text = ""

        size_node = etree.SubElement(root, "size")
        height, width, channels = image.shape
        width_node = etree.SubElement(size_node, "width")
        height_node = etree.SubElement(size_node, "height")
        channels_node = etree.SubElement(size_node, "depth")
        width_node.text = str(width)
        height_node.text = str(height)
        channels_node.text = str(channels)

        seg_node = etree.SubElement(root, "segmented")
        seg_node.text = "0"

        image_status_node = etree.SubElement(root, "image_status")
        image_status_node.text = "3"

        xml_name = prefix + ".xml"

        for i in range(boxes.size(0)):
            box = boxes[i, :]
            object_node = etree.SubElement(root, "object")
            status_node = etree.SubElement(object_node, "status")
            status_node.text = "person"

            name_node = etree.SubElement(object_node, "name")
            name_node.text = "person"
            bndbox_node = etree.SubElement(object_node, "bndbox")
            for i, item in enumerate(["xmin", "ymin", "xmax", "ymax"]):
                item_node = etree.SubElement(bndbox_node, item)
                item_node.text = str(int(box[i]))

        tree = etree.ElementTree(root)
        tree.write(os.path.join(self.save_path, xml_name), pretty_print=True, encoding="utf-8")



