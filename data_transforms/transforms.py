#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq

from tqdm import tqdm

from data_transforms.get_cls_from_name import get_r_cls, get_w_cls, get_p_cls
from data_transforms.reader.reader import ConcatReader
from data_transforms.writer.writer import ConcatWriter
from data_transforms.processor.processor import EmptyProcess

'''
content:
    img:
    info:
        imageName: os.path.split(json_cont["imagePath"])[-1]
        imageWidth: json_cont["imageWidth"]
        imageHeight: json_cont["imageHeight"]
        imageDepth: 
'''


class DataTransforms(object):
    def __init__(self,
                 reader_method, reader_path,
                 writer_method, writer_path,
                 processor=None,
                 ):
        self.reader = self._get_reader(reader_method, reader_path)
        self.writer = self._get_writer(writer_method, writer_path)
        self.processor = self._get_processor(processor)

    def _get_reader(self, reader_method, reader_path):
        readers = []
        if not isinstance(reader_method, (list, tuple)):
            reader_method = (reader_method,)
        if not isinstance(reader_path, (list, tuple)):
            reader_path = (reader_path,)
        for rm, rp in zip(reader_method, reader_path):
            if isinstance(rm, str):
                rm = get_r_cls(rm)
            readers.append(rm(rp))
        return ConcatReader(readers)

    def _get_writer(self, writer_method, writer_path):
        writers = []
        if not isinstance(writer_method, (list, tuple)):
            writer_method = (writer_method,)
        if not isinstance(writer_path, (list, tuple)):
            writer_path = (writer_path,)
        for wm, wp in zip(writer_method, writer_path):
            if isinstance(wm, str):
                wm = get_w_cls(wm)
            writers.append(wm(wp))
        return ConcatWriter(writers)

    def _get_processor(self, processor):
        if isinstance(processor, str):
            processor = get_p_cls(processor)
        if processor is None:
            processor = EmptyProcess
        return processor()

    def apply(self):
        for content in tqdm(self.reader):
            content = self.processor.process(content)
            self.writer.write(content)


if __name__ == '__main__':
    transforms = DataTransforms(reader_method="image",
                                reader_path="test_imgs/inputs/img_or_xml/img/",
                                writer_method="image",
                                writer_path="test_imgs/outputs/img_xml_2_img_vis/",
                                processor="jpg2png",
                                )
    transforms.apply()
