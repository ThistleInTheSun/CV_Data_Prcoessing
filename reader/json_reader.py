#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq


import json
import os

IMG_SUFFIX_LIST = [".jpg", ".jpeg", ".JPG", ".JPEG"]


class JsonReader(object):
    def __init__(self):
        pass

    def load_json(path):
        if os.path.splitext(path)[-1] != ".json":
            print("Cannot find {}".format(path))
            return
        json_file = open(path)
        json_cont = json.load(json_file)
        res = []
        for obj in json_cont["shapes"]:
            cur_obj = {}
            cur_obj["label"] = obj["label"]
            cur_obj["points"] = obj["points"][0] + obj["points"][1]
            res.append(cur_obj)
        json_file.close()
        return res


    def load_json(path):
        if os.path.splitext(path)[-1] != ".json":
            print("Cannot find {}".format(path))
            return
        json_file = open(path)
        json_cont = json.load(json_file)
        res = []
        for obj in json_cont["shapes"]:
            cur_obj = {}
            cur_obj["label"] = obj["label"]
            cur_obj["points"] = obj["points"]
            res.append(cur_obj)
        json_file.close()
        return res
