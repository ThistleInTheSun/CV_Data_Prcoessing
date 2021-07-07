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
    def __init__(self, path):
        self.path = path

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        elif os.listdir(path):
            warn("Writer warning: {} is not empty!".format(path))
        self._path = path

    def __add__(self, other: 'Writer[T_co]') -> 'ConcatWriter[T_co]':
        return ConcatWriter([self, other])

    def write(self, content):
        raise NotImplementedError


class ConcatWriter(object):
    writers: List[Writer[T_co]]

    def __init__(self, writers: Iterable[Writer]) -> None:
        super(ConcatWriter, self).__init__()
        self.writers = list(writers)

    def __len__(self):
        return self.writers.__len__()

    def write(self, content):
        if content is None:
            warn("Writer warning: content is None!")
            return
        for w in self.writers:
            w.write(content)
        return content


class ImageWriter(Writer):
    def __init__(self, path: Text, *args, **kwargs):
        super().__init__(path)
        self.__to_close = False

    def __enter__(self):
        return self

    def write(self, content):
        name = content["info"]["imageName"]
        img = content["image"]
        cv2.imwrite(os.path.join(self.path, name), img)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def close(self):
        self.__to_close = True


class VideoWriter(Writer):
    def __init__(self, path: Text, *args, **kwargs):
        super().__init__(path)
        self.__to_close = False

    def __enter__(self):
        return self

    def write(self, content):
        name = content["info"]["imageName"]
        img = content["image"]
        cv2.imwrite(os.path.join(self.path, name), img)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True

    def close(self):
        self.__to_close = True


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
    classes_dict = {}
    classNums = defaultdict(int)

    def write(self, content):
        name = os.path.splitext(content["info"]["imageName"])[0]
        h, w = content["info"]["imageHeight"], content["info"]["imageWidth"]
        label_txt = open(os.path.join(self.path, name + '.txt'), 'w')
        for obj in content["info"]["shapes"]:
            cls_id = self._get_id(obj["label"])
            if cls_id is None:
                label_txt.close()
                return
            points = self._anno2minmax(obj, w, h)
            label_txt.write(str(cls_id) + " " + " ".join([str(p) for p in points]) + '\n')
        label_txt.close()

    def _get_id(self, label):
        if label not in self.classes_dict:
            self.classes_dict[label] = len(self.classes_dict)
        self.classNums[label] += 1
        cls_id = self.classes_dict[label]
        return cls_id

    def _anno2minmax(self, obj, w, h):
        p0, p1, p2, p3 = obj["points"]  # x1, y1, x2, y2
        x_min = max(min(p0, p2), 0)
        x_max = min(max(p0, p2), w)
        y_min = max(min(p1, p3), 0)
        y_max = min(max(p1, p3), h)
        b = (float(x_min), float(x_max), float(y_min), float(y_max))
        points = self._convert_xxyy2xywh((w, h), b)
        return points

    def _convert_xxyy2xywh(self, size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h
