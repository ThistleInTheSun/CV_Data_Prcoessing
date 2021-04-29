#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq

import os
from typing import *

import cv2

IMAGE_EXTENSIONS = [".mp4", ".avi"]


class VideoReader(object):
    def __init__(self, path: Text, frame_rate=1,
                 color_space: Text = "BGR", *args, **kwargs):
        self.path = path
        self.frameRate = frame_rate
        self.color_space = color_space

        self.idx = 0
        self.cap = cv2.VideoCapture(path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Text):
        if not os.path.exists(path):
            raise ValueError("Reader path is not existed!\n{}".format(path))
        self._path = path

    def __iter__(self):
        return self

    def __next__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.idx % self.frameRate == 0:
                    self.idx += 1
                    return self.cvt(frame)
                self.idx += 1
        else:
            self.cap.release()
            return

    def cvt(self, img):
        if self.color_space == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
