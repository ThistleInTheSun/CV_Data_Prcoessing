import os

import cv2
import numpy as np

__all__ = ["Processor", "EmptyProcess", "DrawProcessor", "Img2JpgProcessor",
           "Img2PngProcessor", "Img2BmpProcessor", "Anno2MaskProcessor"]


class Processor(object):
    def process(self, content):
        raise NotImplementedError


class EmptyProcess(Processor):
    @staticmethod
    def process(x):
        return x


class DrawProcessor(Processor):
    def __init__(self, thickness=2, area_type="box"):
        self.thickness = thickness
        self.area_type = area_type
        self.color_map = {0: (0, 0, 255),
                          1: (0, 255, 0),
                          2: (255, 0, 0),
                          3: (0, 255, 255),
                          }
        self.label_map = {}

    def draw(self, img, points, label):
        color = self.color_map[self.label_map[label]]
        if self.area_type == "polygon":
            points = np.int32([points])
            cv2.polylines(img, points,
                          isClosed=True, color=color, thickness=self.thickness)
        elif self.area_type == "box":
            pt1 = tuple(points[:2])
            pt2 = tuple(points[2:4])
            cv2.rectangle(img=img, pt1=pt1, pt2=pt2,
                          color=color, thickness=self.thickness)
        img = cv2.putText(img, label, tuple(points[:2]), cv2.FONT_HERSHEY_COMPLEX,
                          fontScale=1, color=color, thickness=self.thickness)
        return img

    def join_label(self, label):
        if label not in self.label_map:
            self.label_map[label] = len(self.label_map)

    def process(self, content):
        img, anno = content["image"], content["anno"]
        for obj in anno["shapes"]:
            points = obj["points"]
            label = obj["label"]
            self.join_label(label)
            self.draw(img, points, label)
        content["image"] = img
        return content


class Img2JpgProcessor(Processor):
    def process(self, content):
        img_name = content["info"]["imageName"]
        prefix, suffix = os.path.splitext(img_name)
        img_name = prefix + ".jpg"
        content["info"]["imageName"] = img_name
        return content


class Img2PngProcessor(Processor):
    def process(self, content):
        img_name = content["info"]["imageName"]
        prefix, suffix = os.path.splitext(img_name)
        img_name = prefix + ".png"
        content["info"]["imageName"] = img_name
        return content


class Img2BmpProcessor(Processor):
    def process(self, content):
        img_name = content["info"]["imageName"]
        prefix, suffix = os.path.splitext(img_name)
        img_name = prefix + ".bmp"
        content["info"]["imageName"] = img_name
        return content


class Anno2MaskProcessor(Processor):
    def __init__(self, label_val_dict=None, is_show=False, img_type=None):
        self.label_val_dict = label_val_dict if label_val_dict else {}
        self.is_show = is_show
        self.img_type = img_type

    def process(self, content):
        if "shapes" not in content["info"]:
            raise ValueError("No annotation file to draw mask!")
        img_name = content["info"]["imageName"]
        prefix, suffix = os.path.splitext(img_name)
        img_name = prefix + ".png"
        content["info"]["imageName"] = img_name

        mask = np.zeros(content["image"].shape[:2])
        shapes = content["info"]["shapes"]
        for shape in shapes:
            lable = shape["label"]
            if lable not in self.label_val_dict:
                self.label_val_dict[lable] = len(self.label_val_dict)
                print("new label: {} -> {}".format(lable, self.label_val_dict[lable]))
            points = shape["points"]
            if len(points) == 4 and isinstance(points[0], int):
                pt1 = tuple(points[:2])
                pt2 = tuple(points[2:4])
                mask = cv2.rectangle(img=mask, pt1=pt1, pt2=pt2,
                                     color=self.label_val_dict[lable], thickness=-1)
            else:
                points = np.int32([points])
                mask = cv2.polylines(mask, points, isClosed=True,
                                     color=self.label_val_dict[lable])
        content["img"] = mask
        return content
