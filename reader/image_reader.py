#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/28
# @Author  : xq

import os
from typing import *
import cv2

from manuvision_img_utils.core.io.reader.reader_base import *

__all__ = ["ImageReader"]

IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg", ".bmp"]


class ImageReader(object):
    def __init__(self, directory: Text,
                 color_space: Text = "BGR", *args, **kwargs):
        self.color_space = color_space
        self.directory = directory

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, directory: Text):
        if not os.path.exists(directory):
            raise ValueError("Reader directory is not existed!\n{}".format(directory))
        self._directory = directory
        self.filenames = [item for item in os.listdir(self.directory) if os.path.splitext(item)[-1] in IMAGE_EXTENSIONS]

    def get_file_name(self, item):
        return self.filenames[item]

    def get_file(self, file_name):
        return self.get_image(file_name + ".jpg")

    def __getitem__(self, item):
        return self.get_image(self.filenames[item])

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def autocomplete_suffix(self, filename):
        if filename not in self.filenames:
            for suffix in IMAGE_EXTENSIONS:
                if filename + suffix in self.filenames:
                    filename += suffix
                    break
            else:
                print("not find {}".format(filename))
        return filename

    def get_image(self, filename):
        filename = self.autocomplete_suffix(filename)
        img = cv2.imread(os.path.join(self.directory, filename))
        if self.color_space == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
