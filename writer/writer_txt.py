#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq


import os


def write_train_valid_txt(self):
    train_txt = open(os.path.join(self.target_path, 'train.txt'), 'w')
    for img_path in self.train_list:
        train_txt.write("%s\n" % img_path)
    train_txt.close()

    valid_txt = open(os.path.join(self.target_path, 'valid.txt'), 'w')
    for img_path in self.valid_list:
        valid_txt.write("%s\n" % img_path)
    valid_txt.close()




