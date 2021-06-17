import json
import os
from typing import *
from typing import TypeVar, Generic, Iterable, List
from xml.dom.minidom import parse

import cv2
from manuvision_img_utils.core.io.reader.reader_base import *

IMG_SUFFIX_LIST = [".xml"]
JSON_SUFFIX_LIST = [".json"]

__all__ = ["Reader", "ConcatReader", "ImageReader", "VideoReader", "JsonReader", "XmlReader"]

VIDEO_EXTENSIONS = [".mp4", ".avi"]
IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]

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
            content.update(r.get_file(self.intersection_file_name[idx]))
        return content


class ImageReader(Reader):
    def __init__(self, directory: Text,
                 color_space: Text = "BGR", *args, **kwargs):
        self.color_space = color_space
        self.directory = directory

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, directory: Text):
        if not os.path.exists(directory):
            raise ValueError("Reader directory is not existed!\n{}".format(directory))
        self._directory = directory
        self.filenames = [item for item in os.listdir(self.directory) if os.path.splitext(item)[-1] in IMAGE_EXTENSIONS]

    def get_file_name(self, item):
        return self.filenames[item]

    def get_file(self, file_name):
        return self.get_image(file_name + ".jpg")

    def __getitem__(self, item):
        return self.get_image(self.filenames[item])

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def autocomplete_suffix(self, filename):
        if filename not in self.filenames:
            for suffix in IMAGE_EXTENSIONS:
                if filename + suffix in self.filenames:
                    filename += suffix
                    break
            else:
                print("not find {}".format(filename))
        return filename

    def get_image(self, filename):
        filename = self.autocomplete_suffix(filename)
        img = cv2.imread(os.path.join(self.directory, filename))
        if self.color_space == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        content = {"image": img, "info": {"imageName": filename}}
        return content


class VideoReader(Reader):
    def __init__(self, path: Text, frame_rate=1,
                 color_space: Text = "BGR", *args, **kwargs):
        self.path = path
        self.frameRate = frame_rate
        self.color_space = color_space

        self.idx = 0
        self.cap = cv2.VideoCapture(path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Text):
        if not os.path.exists(path):
            raise ValueError("Reader path is not existed!\n{}".format(path))
        self._path = path

    def __iter__(self):
        return self

    def __next__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.idx % self.frameRate == 0:
                    self.idx += 1
                    return self.cvt(frame)
                self.idx += 1
        else:
            self.cap.release()
            return

    def cvt(self, img):
        if self.color_space == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class JsonReader(Reader):
    def __init__(self, path):
        self.path = path
        self.filenames = [item for item in os.listdir(self.path) if os.path.splitext(item)[-1] in JSON_SUFFIX_LIST]

    def __getitem__(self, item):
        return self.get_json(self.filenames[item])

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def get_json(self, json_name):
        content = {}

        json_file = open(os.path.join(self.path, json_name))
        json_cont = json.load(json_file)
        content["imageData"] = json_cont["imageData"]
        content["imageName"] = os.path.split(json_cont["imagePath"])[-1]
        content["imageWidth"] = json_cont["imageWidth"]
        content["imageHeight"] = json_cont["imageHeight"]
        content["imageDepth"] = json_cont["imageDepth"] if "imageDepth" in json_cont else None
        shapes = []
        for obj in json_cont["shapes"]:
            cur_obj = {"label": obj["label"],
                       "line_color": obj["line_color"],
                       "fill_color": obj["fill_color"],
                       "shape_type": obj["shape_type"],
                       "points": obj["points"],
                       }

            shapes.append(cur_obj)
        json_file.close()
        content["shapes"] = shapes
        return content


class XmlReader(Reader):
    def __init__(self, path):
        self.path = path
        self.filenames = [item for item in os.listdir(self.path) if os.path.splitext(item)[-1] in IMG_SUFFIX_LIST]

    def get_file_name(self, item):
        return self.filenames[item]

    def get_file(self, file_name):
        return self._get_xml(file_name + ".xml")

    def __getitem__(self, item):
        return self._get_xml(self.filenames[item])

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def _get_xml(self, path):
        if os.path.splitext(path)[-1] != ".xml":
            print("Cannot find {}".format(path))
            return
        content = {}
        dom_tree = parse(os.path.join(self.path, path))
        root_node = dom_tree.documentElement
        content["imageName"] = root_node.getElementsByTagName("filename")[0].childNodes[0].data
        size_node = root_node.getElementsByTagName("size")[0]
        content["imageWidth"] = size_node.getElementsByTagName("width")[0].childNodes[0].data
        content["imageHeight"] = size_node.getElementsByTagName("height")[0].childNodes[0].data
        content["imageDepth"] = size_node.getElementsByTagName("depth")[0].childNodes[0].data

        objects = root_node.getElementsByTagName("object")
        shapes = []
        for obj in objects:
            cur_obj = dict()
            cur_obj["label"] = obj.getElementsByTagName("name")[0].childNodes[0].data
            cur_obj["points"] = [
                int(obj.getElementsByTagName("bndbox")[0].getElementsByTagName(point)[0].childNodes[0].data) \
                for point in ["xmin", "ymin", "xmax", "ymax"]]
            shapes.append(cur_obj)
        content["shapes"] = shapes
        return content
