#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq


import os
from xml.dom.minidom import parse

IMG_SUFFIX_LIST = [".xml"]


class XmlReader(object):
    def __init__(self, path):
        self.path = path
        self.filenames = [item for item in os.listdir(self.path) if os.path.splitext(item)[-1] in IMG_SUFFIX_LIST]

    def get_file_name(self, item):
        return self.filenames[item]

    def get_file(self, file_name):
        return self._get_xml(file_name + ".xml")

    def __getitem__(self, item):
        return self._get_xml(self.filenames[item])

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def _get_xml(self, path):
        if os.path.splitext(path)[-1] != ".xml":
            print("Cannot find {}".format(path))
            return
        content = {}
        dom_tree = parse(os.path.join(self.path, path))
        root_node = dom_tree.documentElement
        content["imageName"] = root_node.getElementsByTagName("filename")[0].childNodes[0].data
        size_node = root_node.getElementsByTagName("size")[0]
        content["imageWidth"] = size_node.getElementsByTagName("width")[0].childNodes[0].data
        content["imageHeight"] = size_node.getElementsByTagName("height")[0].childNodes[0].data
        content["imageDepth"] = size_node.getElementsByTagName("depth")[0].childNodes[0].data

        objects = root_node.getElementsByTagName("object")
        shapes = []
        for obj in objects:
            cur_obj = dict()
            cur_obj["label"] = obj.getElementsByTagName("name")[0].childNodes[0].data
            cur_obj["points"] = [
                int(obj.getElementsByTagName("bndbox")[0].getElementsByTagName(point)[0].childNodes[0].data) \
                for point in ["xmin", "ymin", "xmax", "ymax"]]
            shapes.append(cur_obj)
        content["shapes"] = shapes
        return content
