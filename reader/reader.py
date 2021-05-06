#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/29
# @Author  : xq

from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple
import os

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Reader(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Reader[T_co]') -> 'ConcatReader[T_co]':
        return ConcatReader([self, other])


class ConcatReader(object):
    r"""Reader as a concatenation of multiple readers.

    This class is useful to assemble different existing readers.

    Args:
        readers (sequence): List of readers to be concatenated
    """
    readers: List[Reader[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        set_list = []
        for r in sequence:
            filenames = r.filenames
            r_set = set([os.path.splitext(x)[0] for x in filenames])
            set_list.append(r_set)
        set0 = set_list[0]
        for i in set_list:
            set0 = set0 & i
        return list(set0)

    def __init__(self, readers: Iterable[Reader]) -> None:
        super(ConcatReader, self).__init__()
        # Cannot verify that readers is Sized
        assert len(readers) > 0, 'readers should not be an empty iterable'  # type: ignore
        self.readers = list(readers)
        # for r in self.readers:
        #     assert not isinstance(r, IterableReader), "ConcatDataset does not support IterableDataset"
        self.intersection_file_name = self.cumsum(self.readers)

    def __len__(self):
        return len(self.intersection_file_name)

    def __getitem__(self, idx):
        content = {}
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        for r in self.readers:
            r_name = r.__class__.__name__
            if "Xml" in r_name or "Json" in r_name:
                content["anno"] = r.get_file(self.intersection_file_name[idx])
            else:
                content["image"] = r.get_file(self.intersection_file_name[idx])
        return content
