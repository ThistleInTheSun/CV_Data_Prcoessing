#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq

from get_cls_from_name import get_r_cls, get_w_cls


class DataTransforms(object):
    def __init__(self,
                 reader_method, reader_path,
                 writer_method, writer_path,
                 processor=None,
                 ):
        self.reader = self._get_reader(reader_method, reader_path)
        self.writer = self._get_writer(writer_method, writer_path)
        self.processor = processor

    def _get_reader(self, reader_method, reader_path):
        readers = []
        for rm, rp in zip(reader_method, reader_path):
            if isinstance(rm, str):
                rm = get_r_cls(rm)
            readers.append(rm(rp))
        return ConcatReader(readers)

    def _get_writer(self, writer_method, writer_path):
        writers = []
        if "img" in writer_method or "image" in writer_method:
            writers.append(ImageWriter)
        if "video" in writer_method:
            writers.append(VideoWriter)
        return ConcatWriter(writers)

    def apply(self):
        content = self.reader.read()
        content = self.processor.process(content)
        self.writer.write(content)


if __name__ == '__main__':
    transforms = DataTransforms(reader_method=("video", "xml"),
                                reader_path=("path1", "path2"),
                                writer_method=("image", "json"),
                                writer_path=("path1", "path2"),
                                )




