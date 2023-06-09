#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq

from tqdm import tqdm

from cvdata.core.get_cls_from_name import get_r_cls, get_w_cls, get_p_cls
from cvdata.core.processor import EmptyProcess
from cvdata.core.reader import ConcatReader
from cvdata.core.writer import ConcatWriter

'''
content:
    img:
    info:
        imageName:
        imageWidth:
        imageHeight:
        imageDepth: 
        shapes:[
            {
                shape_type: "polygon",  # [polygon | bndbox]
                label: str,
                points': [(693, 224), (739, 253), ...],  # point1, point2, ...
            },
            {
                'shape_type': "bndbox",
                'label': 'hand', 
                'points': [692, 254, 709, 275],  # x_min, y_min, x_max, y_max.
            },
            ...
        ]
'''


class DataTransforms(object):
    def __init__(self,
                 reader_method, writer_method,
                 processor=None, is_recursive=False,
                 ):
        print("reading:", reader_method)
        print("writing:", writer_method)
        self.is_recursive = is_recursive
        self.reader = self._get_reader(reader_method)
        self.writer = self._get_writer(writer_method)
        self.processors = self._get_processors(processor)

    def _get_reader(self, reader_method):
        readers = []
        for rm, rp in reader_method.items():
            if isinstance(rm, str):
                rm = get_r_cls(rm)
            readers.append(rm(rp))
        return ConcatReader(readers, self.is_recursive)

    def _get_writer(self, writer_method):
        writers = []
        if isinstance(writer_method, dict):
            for wm, wp in writer_method.items():
                if isinstance(wm, str):
                    wm = get_w_cls(wm)
                writers.append(wm(wp))
        elif isinstance(writer_method, list):
            for wm in writer_method:
                writers.append(wm)
        else:
            writers.append(writer_method)
        return ConcatWriter(writers)

    def _get_processors(self, processor):
        processors = []
        if not isinstance(processor, list or tuple):
            processor = [processor]
        for p in processor:
            if p is None or p in ["none", "None"]:
                p = EmptyProcess()
            if isinstance(p, str):
                p = get_p_cls(p)()
            processors.append(p)
        return processors

    def apply(self):
        for content in tqdm(self.reader):
            for p in self.processors:
                content = p.process(content)
            self.writer.write(content)
        self.writer.close()


if __name__ == '__main__':
    class CropProcessor:
        def process(self, content):
            content["image"] = content["image"][100: 500, 200: 600]
            return content

    transforms = DataTransforms(reader_method={"name": "../test_imgs/inputs",
                                               # "json": "../test_imgs/inputs/img_and_json",
                                               },
                                writer_method={"name": "../test_imgs/outputs/",
                                               },
                                processor=None,
                                is_recursive=True,
                                )
    transforms.apply()
