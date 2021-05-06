#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq

import os
import threading
from queue import Queue, Empty, Full
from typing import *
from warnings import warn

import cv2
import numpy as np

__all__ = ["ImageWriter"]


class ImageWriter(object):
    """Writer Class.

    Usage:
    1:
        writer = Writer("Your_target_directory")
        writer.save(img, filename)
        writer.close()
    2:
        with Writer("Your_target_directory") as writer:
            writer.save(img, filename)
    """

    def __init__(self, directory: Text, suffix=None, *args, **kwargs):
        """Init

        :param directory:
        :param max_size: Max size of queue.
        :param _:
        :param timeout:
        :param kwargs:
        """
        super().__init__()
        self.directory = directory
        self.suffix = suffix
        self.__to_close = False

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            # warn("Writer warning: {} is not existed, created.".format(directory))
        elif os.listdir(directory):
            warn("Writer warning: {} is not empty!".format(directory))
        # os.makedirs(directory, exist_ok=True)
        self._directory = directory

    def __enter__(self):
        return self

    def write(self, content):
        filename = content["anno"]["imageName"]
        img = content["image"]
        cv2.imwrite(os.path.join(self.directory, filename), img)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def close(self):
        self.__to_close = True
