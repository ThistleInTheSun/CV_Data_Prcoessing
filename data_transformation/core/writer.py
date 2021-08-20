import os
import random
from collections import defaultdict
from typing import *
from typing import TypeVar, Generic, Iterable, List
from warnings import warn

import cv2
from lxml import etree

__all__ = ["Writer", "ConcatWriter", "ImageWriter", "VideoWriter",
           "JsonWriter", "XmlWriter", "TxtWriter", "NameWriter"]
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Writer(Generic[T_co]):
    def __init__(self, root):
        self.root = root
        self.update_path()

    def update_path(self, ptype=""):
        path = os.path.join(self.root, ptype)
        if not os.path.exists(path):
            os.makedirs(path)
        # elif os.listdir(path):
        #     warn("Writer warning: {} is not empty!".format(path))
        self.path = path

    def __add__(self, other: 'Writer[T_co]') -> 'ConcatWriter[T_co]':
        return ConcatWriter([self, other])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, content):
        raise NotImplementedError

    def close(self):
        pass


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
        ptype = content["info"]["ptype"]
        for w in self.writers:
            w.update_path(ptype)
            w.write(content)
        return content

    def close(self):
        for w in self.writers:
            w.close()


class ImageWriter(Writer):
    def write(self, content):
        name = content["info"]["imageName"]
        img = content["image"]
        cv2.imwrite(os.path.join(self.path, name), img)


class VideoWriter(Writer):
    def __init__(self, path: Text, video_name=None, video_fps=25, img_size=None):
        super().__init__(path)
        self.video_name = video_name
        self.video_fps = video_fps
        self.img_size = img_size
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.img_size and self.video_name:
            self.save_path = os.path.join(self.path, self.video_name + ".mp4")
            self.video = cv2.VideoWriter(self.save_path, self.fourcc, self.video_fps, self.img_size)

    def write(self, content):
        if not content or "image" not in content:
            return {}
        if not self.img_size:
            self.img_size = content["image"].shape[:2][::-1]
            name = os.path.splitext(content["info"]["imageName"])[0].rsplit("_", 1)[0]
            self.save_path = os.path.join(self.path, name + ".mp4")
            self.video = cv2.VideoWriter(self.save_path, self.fourcc, self.video_fps, self.img_size)
        img = content["image"]
        self.video.write(img)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.video.release()
        return True

    def close(self):
        self.video.release()


class JsonWriter(Writer):
    def write(self, content):
        pass


class XmlWriter(Writer):
    def write(self, content):
        name = content["info"]["imageName"]
        prefix = name.split('.jpg')[0]
        root = etree.Element("annotation")
        file_name = etree.SubElement(root, "filename")
        file_name.text = prefix
        url = etree.SubElement(root, "url")
        url.text = ""

        size_node = etree.SubElement(root, "size")
        height = content["info"]["imageHeight"]
        width = content["info"]["imageWidth"]
        channels = content["info"]["imageDepth"]
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

        for shape in content["info"]["shapes"]:
            points = shape["points"]
            object_node = etree.SubElement(root, "object")
            status_node = etree.SubElement(object_node, "status")
            status_node.text = "person"

            name_node = etree.SubElement(object_node, "name")
            name_node.text = "person"

            if len(points) == 4 and not isinstance(points[0], Iterable):
                node = etree.SubElement(object_node, "bndbox")
                for i, item in enumerate(["xmin", "ymin", "xmax", "ymax"]):
                    item_node = etree.SubElement(node, item)
                    item_node.text = str(int(points[i]))
            else:
                node = etree.SubElement(object_node, "polygon")
                for p in points:
                    shape_node = etree.SubElement(node, "point")
                    for i, item in enumerate(["x", "y"]):
                        item_node = etree.SubElement(shape_node, item)
                        item_node.text = str(int(p[i]))

        tree = etree.ElementTree(root)
        tree.write(os.path.join(self.path, xml_name), pretty_print=True, encoding="utf-8")


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


class NameWriter(Writer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_txt = open(os.path.join(self.root, 'train.txt'), 'w')
        self.val_txt = open(os.path.join(self.root, 'val.txt'), 'w')
        self.test_txt = open(os.path.join(self.root, 'test.txt'), 'w')
        self.train_rate = 0.8
        self.val_rate = (1 - self.train_rate) / 2
        self.test_rate = 1 - self.train_rate - self.val_rate

    def write(self, content):
        info = content["info"]
        if "info" not in content \
                or "imageName" not in content["info"] \
                or not content["info"]["imageName"]:
            return
        line = os.path.join(info["ptype"], info["imageName"])
        ran = random.random()
        if ran <= self.train_rate:
            self.train_txt.write(line + '\n')
        elif ran <= self.train_rate + self.val_rate:
            self.val_txt.write(line + '\n')
        else:
            self.test_txt.write(line + '\n')

    def close(self):
        self.train_txt.close()
        self.val_txt.close()
        self.test_txt.close()
