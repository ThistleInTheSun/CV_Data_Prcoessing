#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq

from tqdm import tqdm

from get_cls_from_name import get_r_cls, get_w_cls, get_p_cls
from reader.reader import ConcatReader
from writer.writer import ConcatWriter
from processor.draw_processor import EmptyProcess


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
        for rm, rp in zip(reader_method, reader_path):
            if isinstance(rm, str):
                rm = get_r_cls(rm)
            readers.append(rm(rp))
        return ConcatReader(readers)

    def _get_writer(self, writer_method, writer_path):
        writers = []
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
    # transforms = DataTransforms(reader_method=("image", "json"),
    #                             reader_path=("test_imgs/inputs/img_and_json/", "test_imgs/inputs/img_and_json/"),
    #                             writer_method=("image", "xml"),
    #                             writer_path=("test_imgs/outputs/img_and_xml/", "test_imgs/outputs/img_and_xml/"),
    #                             )
    transforms = DataTransforms(reader_method=("image", "xml",),
                                reader_path=("test_imgs/inputs/img_or_xml/img/", "test_imgs/inputs/img_or_xml/xml/",),
                                writer_method=("image",),
                                writer_path=("test_imgs/outputs/img_xml_2_img_vis/",),
                                processor="draw",
                                )
    transforms.apply()
