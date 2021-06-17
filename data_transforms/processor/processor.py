import os

import cv2
import numpy as np


class Processor(object):
    def process(self, content):
        raise NotImplementedError


class EmptyProcess(Processor):
    @staticmethod
    def process(x):
        return x


class DrawProcessor(Processor):
    def __init__(self, color=(0, 0, 255), thickness=2, area_type="box"):
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
        # pil_img = Image.fromarray(img)
        # visualize.draw_texts(pil_img, [points[:2]], [label], color="red", back_color=(0, 0, 255),
        #                      font_size=24,
        #                      position="manu", offset=(0, -24 - 10))
        # img = np.asarray(pil_img)
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


class Jpg2PngProcessor(Processor):
    def process(self, content):
        img_name = content["info"]["imageName"]
        prefix, suffix = os.path.splitext(img_name)
        if suffix == ".jpg":
            img_name = prefix + ".png"
        content["info"]["imageName"] = img_name
        return content
