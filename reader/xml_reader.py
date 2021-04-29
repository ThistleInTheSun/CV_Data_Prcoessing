#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq


import os
from xml.dom.minidom import parse

IMG_SUFFIX_LIST = [".jpg", ".jpeg", ".JPG", ".JPEG"]


class XmlReader(object):
    def __init__(self):
        pass

    def load_xml(path):
        if os.path.splitext(path)[-1] != ".xml":
            print("Cannot find {}".format(path))
            return
        res = []
        dom_tree = parse(path)
        root_node = dom_tree.documentElement
        objects = root_node.getElementsByTagName("object")
        for obj in objects:
            cur_obj = dict()
            cur_obj["label"] = obj.getElementsByTagName("name")[0].childNodes[0].data
            cur_obj["points"] = [
                int(obj.getElementsByTagName("bndbox")[0].getElementsByTagName(point)[0].childNodes[0].data) \
                for point in ["xmin", "ymin", "xmax", "ymax"]]
            res.append(cur_obj)
        return res
