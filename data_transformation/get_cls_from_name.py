#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq

import importlib


def get_cls(obj_name, r_or_w, *args, **kwargs):
    """Import the module "models/[model_name].py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    obj_name = obj_name + "_" + r_or_w
    modellib = importlib.import_module("data_transformation." + r_or_w)
    model = None
    target_obj_name = obj_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_obj_name.lower():
            # and issubclass(cls, base_model):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase."
              % (modellib, target_obj_name))
        exit(0)

    return model


def get_r_cls(*args, **kwargs):
    return get_cls(r_or_w="reader", *args, **kwargs)


def get_w_cls(*args, **kwargs):
    return get_cls(r_or_w="writer", *args, **kwargs)


def get_p_cls(*args, **kwargs):
    return get_cls(r_or_w="processor", *args, **kwargs)


if __name__ == '__main__':
    get_r_cls(obj_name="xml")
    get_w_cls(obj_name="xml")
