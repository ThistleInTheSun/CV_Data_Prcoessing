import bisect
import json
import os
import warnings
from typing import *
from typing import TypeVar, Generic, Iterable, List
from xml.dom.minidom import parse
import warnings

import cv2

__all__ = ["Reader", "ConcatReader", "ImageReader", "VideoReader", "JsonReader", "XmlReader"]

IMAGE_SUFFIX_LIST = [".jpg", ".png", ".bmp", ".jpeg", ".tiff"]
VIDEO_SUFFIX_LIST = [".mp4", ".avi"]
JSON_SUFFIX_LIST = [".json"]
XML_SUFFIX_LIST = [".xml"]

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Reader(Generic[T_co]):
    def __init__(self, root, ptype="", suffix_list=None, *args, **kwargs):
        self.root = root
        self.ptype = ptype
        self.path = os.path.join(root, ptype)
        self.suffix_list = suffix_list
        self._filenames = None
        self._folders = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Text):
        if not os.path.exists(path):
            raise ValueError("Reader path is not existed!\n{}".format(path))
        self._path = path

    @property
    def filenames(self):
        if self._filenames is None:
            self._filenames = [item for item in os.listdir(self.path)
                               if os.path.splitext(item)[-1] in self.suffix_list]
        return self._filenames

    @property
    def folders(self):
        if self._folders is None:
            self._folders = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
        return self._folders

    def __getitem__(self, item) -> T_co:
        return self.get_content(self.filenames[item])

    def __iter__(self):
        return iter(self[i] for i in range(len(self)))

    def __len__(self):
        return len(self.filenames)

    def get_name(self, item):
        return self.filenames[item]

    def autocomplete_suffix(self, name):
        if name not in self.filenames:
            for suffix in self.suffix_list:
                if name + suffix in self.filenames:
                    name += suffix
                    break
            else:
                print("not find {}".format(name))
        return name

    def get_file(self, name):
        name = self.autocomplete_suffix(name)
        content = self.get_content(name)
        content["info"]["ptype"] = self.ptype
        return content

    def get_content(self, name):
        print(self.__dir__, "NotImplementedError")
        raise NotImplementedError


class ConcatReader(object):
    readers: List[Reader[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def intersect(sequence):
        if len(sequence) == 1:
            return sequence[0]
        set_list = []
        for r in sequence:
            filenames = r.filenames
            r_set = set([os.path.splitext(x)[0] for x in filenames])
            set_list.append(r_set)
        set0 = set_list[0]
        for i in set_list:
            set0 = set0 & i
        return list(set0)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, readers: Iterable[Reader], is_recursive=False) -> None:
        super(ConcatReader, self).__init__()
        self.readers = readers
        self.is_recursive = is_recursive
        self.intersection_file_name = self.intersect([r.filenames for r in self.readers])
        if is_recursive:
            sub_folders = self.intersect([r.folders for r in self.readers])
            self.sub_concat_readers = self._get_sub_concat_reader(readers, sub_folders, self.is_recursive)
            self.cumulative_sizes = self.cumsum(self.sub_concat_readers)

    @property
    def readers(self):
        return self._readers

    @readers.setter
    def readers(self, readers):
        """When __getitem__, overwrite the previous anno info with the following image info in:
            imageName, imageWidth, imageHeight, imageDepth.
        """
        head, tail = [], []
        for r in readers:
            if isinstance(r, ImageReader) or isinstance(r, VideoReader):
                tail.append(r)
            else:
                head.append(r)
        self._readers = head + tail

    @classmethod
    def _get_sub_concat_reader(cls, readers, sub_folders, is_recursive):
        sub_c_readers = []
        for f in sub_folders:
            sub_readers = [globals()[type(r).__name__](root=r.root, ptype=os.path.join(r.ptype, f)) for r in readers]
            sub_c_readers.append(cls(sub_readers, is_recursive))
        return sub_c_readers

    def __iter__(self):
        return iter(self[i] for i in range(len(self)))

    def __len__(self):
        if isinstance(self.readers[0], SubVideoReader):
            # 默认读视频时有且仅有一个video_reader
            l = len(self.readers[0]) if not self.is_recursive \
                else len(self.readers[0]) + sum([len(d) for d in self.sub_concat_readers])
        else:
            l = len(self.intersection_file_name) if not self.is_recursive \
                else len(self.intersection_file_name) + sum([len(d) for d in self.sub_concat_readers])
        return l

    def __getitem__(self, idx):
        content = {"image": None, "info": {}}
        if idx < len(self.intersection_file_name):
            for r in self.readers:
                name = self.intersection_file_name[idx]
                new_content = r.get_file(name)
                content["info"].update(new_content["info"])
                content["image"] = new_content["image"]
            return content
        elif self.is_recursive:
            idx = idx - len(self.intersection_file_name)
            reader_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if reader_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[reader_idx - 1]
            return self.sub_concat_readers[reader_idx][sample_idx]


class ImageReader(Reader):
    def __init__(self, root: Text, ptype="", *args, **kwargs):
        super(ImageReader, self).__init__(root, ptype, IMAGE_SUFFIX_LIST, *args, **kwargs)

    def get_content(self, name):
        name = self.autocomplete_suffix(name)
        img = cv2.imread(os.path.join(self.path, name))
        h, w = img.shape[:2]
        content = {"image": img,
                   "info": {"imageName": name,
                            "imageWidth": w,
                            "imageHeight": h,
                            "imageDepth": 3 if len(img.shape) > 2 and img.shape[2] == 3 else 1
                            }
                   }
        return content


class VideoReader(Reader):
    def __init__(self, path: Text, frame_rate=1, *args, **kwargs):
        super(VideoReader, self).__init__(path, VIDEO_SUFFIX_LIST, *args, **kwargs)
        self.frameRate = frame_rate

        self.video_name = os.path.splitext(os.path.split(path)[1])[0]
        self.idx = 0
        self.cap = cv2.VideoCapture(path)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        return self

    def __len__(self):
        return self.length // self.frameRate

    def __getitem__(self, item) -> T_co:
        return self.__next__()

    def __next__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.idx % self.frameRate == 0:
                    self.idx += 1
                    return self._gen_content(frame, self.idx)
                self.idx += 1
        else:
            self.cap.release()
            return

    def get_file(self, content):
        return content

    def _gen_content(self, img, idx):
        name = self.video_name + str(idx).rjust(6, "0") + ".jpg"
        content = {"image": img,
                   "info": {"imageName": name,
                            "imageWidth": self.width,
                            "imageHeight": self.height,
                            "imageDepth": 3 if len(img.shape) > 2 and img.shape[2] == 3 else 1
                            }
                   }
        return content


class JsonReader(Reader):
    def __init__(self, root, ptype="", *args, **kwargs):
        super(JsonReader, self).__init__(root, ptype, JSON_SUFFIX_LIST, *args, **kwargs)

    def get_content(self, json_name):
        info = {}
        json_file = open(os.path.join(self.path, json_name))
        json_cont = json.load(json_file)
        info["imageData"] = json_cont["imageData"]
        info["imageWidth"] = json_cont["imageWidth"]
        info["imageHeight"] = json_cont["imageHeight"]
        info["imageDepth"] = json_cont["imageDepth"] if "imageDepth" in json_cont else None
        shapes = []
        for obj in json_cont["shapes"]:
            if obj["shape_type"] == "polygon":
                points = [(x, y) for x, y in obj["points"]]
            elif obj["shape_type"] in ["box", "bndbox"]:
                points = obj["points"][0] + obj["points"][1]
                points = [points[0], points[2], points[1], points[3]]
            else:
                points = obj["points"]
                warnings.warn("Unknown type {}: {}.".format(obj["shape_type"], obj["points"]))

            cur_obj = {"label": obj["label"],
                       "line_color": obj["line_color"],
                       "fill_color": obj["fill_color"],
                       "shape_type": obj["shape_type"],
                       "points": points,
                       }
            shapes.append(cur_obj)
        json_file.close()
        info["shapes"] = shapes
        content = {"info": info, "image": None}
        return content


class XmlReader(Reader):
    def __init__(self, root, ptype="", *args, **kwargs):
        super(XmlReader, self).__init__(root, ptype, XML_SUFFIX_LIST, *args, **kwargs)

    def get_content(self, xml_name):
        info = {}
        dom_tree = parse(os.path.join(self.path, xml_name))
        root_node = dom_tree.documentElement
        size_node = root_node.getElementsByTagName("size")[0]
        info["imageWidth"] = int(size_node.getElementsByTagName("width")[0].childNodes[0].data)
        info["imageHeight"] = int(size_node.getElementsByTagName("height")[0].childNodes[0].data)
        info["imageDepth"] = int(size_node.getElementsByTagName("depth")[0].childNodes[0].data)

        objects = root_node.getElementsByTagName("object")
        shapes = []
        for obj in objects:
            cur_obj = {"label": obj.getElementsByTagName("name")[0].childNodes[0].data,
                       "points": [],
                       }
            if obj.getElementsByTagName("bndbox"):
                cur_obj["shape_type"] = "bndbox"
                for point in ["xmin", "ymin", "xmax", "ymax"]:
                    cur_obj["points"].append(
                        int(obj.getElementsByTagName("bndbox")[0].getElementsByTagName(point)[0].childNodes[0].data))
            elif obj.getElementsByTagName("polygon"):
                cur_obj["shape_type"] = "polygon"
                for point in obj.getElementsByTagName("polygon")[0].getElementsByTagName("point"):
                    x = int(float(point.getElementsByTagName("x")[0].childNodes[0].data))
                    y = int(float(point.getElementsByTagName("y")[0].childNodes[0].data))
                    cur_obj["points"].append((x, y))
            shapes.append(cur_obj)
        info["shapes"] = shapes
        content = {"info": info, "image": None}
        return content
