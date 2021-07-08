#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq

from tqdm import tqdm

from data_transforms.get_cls_from_name import get_r_cls, get_w_cls, get_p_cls
from data_transforms.processor.processor import EmptyProcess
from data_transforms.reader.reader import ConcatReader
from data_transforms.writer.writer import ConcatWriter

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
                points': [(693, 224), (739, 253), ...],
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
        self.reader = self._get_reader(reader_method)
        self.writer = self._get_writer(writer_method)
        self.processor = self._get_processor(processor)
        self.is_recursive = is_recursive

    def _get_reader(self, reader_method):
        readers = []
        for rm, rp in reader_method.items():
            if isinstance(rm, str):
                rm = get_r_cls(rm)
            readers.append(rm(rp))
        return ConcatReader(readers, self.is_recursive)

    def _get_writer(self, writer_method):
        writers = []
        for wm, wp in writer_method.items():
            if isinstance(wm, str):
                wm = get_w_cls(wm)
            writers.append(wm(wp))
        return ConcatWriter(writers)

    def _get_processor(self, processor):
        if isinstance(processor, str):
            processor = get_p_cls(processor)()
        if processor is None:
            processor = EmptyProcess()
        return processor

    def apply(self):
        for content in tqdm(self.reader):
            content = self.processor.process(content)
            self.writer.write(content)
        self.writer.close()


if __name__ == '__main__':
    transforms = DataTransforms(reader_method={"image": "/home/xq/文档/projects/日联/data",
                                               "xml": "/home/xq/文档/projects/日联/data",
                                               },
                                writer_method={"image": "/home/xq/文档/projects/日联/data_jpg",
                                               "json": "/home/xq/文档/projects/日联/data",
                                               },
                                processor="crop",
                                is_recursive=False,
                                )
    transforms.apply()
