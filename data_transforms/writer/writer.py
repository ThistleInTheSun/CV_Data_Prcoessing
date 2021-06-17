import os
from typing import *
from typing import TypeVar, Generic, Iterable, List
from warnings import warn

import cv2
from lxml import etree

__all__ = ["Writer", "ConcatWriter", "ImageWriter", "VideoWriter", "JsonWriter", "XmlWriter", "TxtWriter"]
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
        return content


class ImageWriter(Writer):
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
        filename = content["info"]["imageName"]
        img = content["image"]
        cv2.imwrite(os.path.join(self.directory, filename), img)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def close(self):
        self.__to_close = True


class VideoWriter(Writer):
    pass


class JsonWriter(Writer):
    def __init__(self, path):
        self.path = path


class XmlWriter(Writer):
    def __init__(self, path):
        self.save_path = path

    def write(self, content):
        name = content["imageName"]
        prefix = name.split('.jpg')[0]
        root = etree.Element("annotation")
        file_name = etree.SubElement(root, "filename")
        file_name.text = prefix
        url = etree.SubElement(root, "url")
        url.text = ""

        size_node = etree.SubElement(root, "size")
        height = content["imageHeight"]
        width = content["imageWidth"]
        channels = content["imageDepth"]
        width_node = etree.SubElement(size_node, "width")
        height_node = etree.SubElement(size_node, "height")
        channels_node = etree.SubElement(size_node, "depth")
        width_node.text = str(width)
        height_node.text = str(height)
        channels_node.text = str(channels)

        seg_node = etree.SubElement(root, "segmented")
        seg_node.text = "0"

        image_status_node = etree.SubElement(root, "image_status")
        image_status_node.text = "3"

        xml_name = prefix + ".xml"

        for i in range(boxes.size(0)):
            box = boxes[i, :]
            object_node = etree.SubElement(root, "object")
            status_node = etree.SubElement(object_node, "status")
            status_node.text = "person"

            name_node = etree.SubElement(object_node, "name")
            name_node.text = "person"
            bndbox_node = etree.SubElement(object_node, "bndbox")
            for i, item in enumerate(["xmin", "ymin", "xmax", "ymax"]):
                item_node = etree.SubElement(bndbox_node, item)
                item_node.text = str(int(box[i]))

        tree = etree.ElementTree(root)
        tree.write(os.path.join(self.save_path, xml_name), pretty_print=True, encoding="utf-8")


class TxtWriter(Writer):
    def write_train_valid_txt(self):
        train_txt = open(os.path.join(self.target_path, 'train.txt'), 'w')
        for img_path in self.train_list:
            train_txt.write("%s\n" % img_path)
        train_txt.close()

        valid_txt = open(os.path.join(self.target_path, 'valid.txt'), 'w')
        for img_path in self.valid_list:
            valid_txt.write("%s\n" % img_path)
        valid_txt.close()
