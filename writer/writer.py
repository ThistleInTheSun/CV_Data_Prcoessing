#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq


from typing import TypeVar, Generic, Iterable, List

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Writer(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Writer[T_co]') -> 'ConcatWriter[T_co]':
        return ConcatWriter([self, other])


class ConcatWriter(object):
    writers: List[Writer[T_co]]

    def __init__(self, writers: Iterable[Writer]) -> None:
        super(ConcatWriter, self).__init__()
        self.writers = list(writers)

    def __len__(self):
        return self.writers.__len__()

    def write(self, content):
        for w in self.writers:
            w_name = w.__class__.__name__
            w.write(content)
            # if "Xml" in w_name:
            #     w.write(content["anno"])
            # elif "Json" in w_name:
            #     w.write(content["anno"])
            # else:
            #     w.write(content["image"])
        return content
