# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import *
from xml.dom.minidom import parse
from PIL import ImageFont, ImageDraw, Image
from collections import defaultdict


__all__ = ["Image2Video"]


def add_text(img, text, test_size, location, color):
    font_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-SemiBold.ttc"
    b, g, r = color
    a = 0
    font = ImageFont.truetype(font_path, test_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(location, text, font=font, fill=(b, g, r, a))
    img = np.array(img_pil)
    return img


class Image2Video(object):
    def __init__(self,
                 img_path: Text,
                 video_path: Text,
                 video_fps: int,
                 test_color=(0, 0, 255)):
        self.img_path = img_path
        self.video_path = video_path
        self.video_fps = video_fps
        self.text_color = test_color

        self.img_size = self.get_img_size()
        self.location = (int(self.img_size[0] * 0.7), int(self.img_size[1] * 0.1))
        self.obj_num = []
        self.color_list = [(0, 255, 0),
                           (50, 255, 50),
                           (50, 20, 255),
                           ]
        self.color_dic = {}

    def get_img_size(self):
        file_list = sorted(os.listdir(self.img_path))
        img0 = cv2.imread(os.path.join(self.img_path, file_list[0]))
        return img0.shape[:2][::-1]

    def _ol_xml_2_mask(self,
                       xml_path: Text,
                       img: np.ndarray):
        xml_file = parse(xml_path)
        annotation = xml_file.documentElement
        object_list = annotation.getElementsByTagName("object")
        current_obj_num = defaultdict(int)
        for obj in object_list:
            name = obj.getElementsByTagName("name")[0].childNodes[0].data
            if name not in self.obj_num:
                self.obj_num.append(name)
            current_obj_num[name] += 1
            if name in self.color_dic:
                color = self.color_dic[name]
            elif self.color_list:
                color = self.color_list[0]
                self.color_list.pop(0)
            else:
                color = np.random.randint(0, 256, 3)
            self.color_dic[name] = color

            polygon = obj.getElementsByTagName("polygon")[0]
            pt_list = polygon.getElementsByTagName("pt")
            point_list = []
            for pt in pt_list:
                x = pt.getElementsByTagName("x")[0].childNodes[0].data
                y = pt.getElementsByTagName("y")[0].childNodes[0].data
                point_list.append([int(float(x)), int(float(y))])
            p1, p2 = point_list[0], point_list[2]
            cv2.rectangle(img, (p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]), color, 1)
        text = "老鼠数量: " + str(current_obj_num["mouse"]) + "\n"
        # text = ""
        # for key in self.obj_num:
        #     # text += "num of {}: ".format(str(key))
        #     text += str(current_obj_num[key]) if current_obj_num else str(0)
        #     text += "\n"
        test_size = int(min(self.img_size) * 0.05)
        img = add_text(img, text, test_size, self.location, self.text_color)
        return img

    def apply(self):
        file_list = sorted(os.listdir(self.img_path))
        save_path = os.path.join(self.video_path, "video" + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(save_path, fourcc, self.video_fps, self.img_size)

        for file_name in tqdm(file_list):
            if not file_name.endswith('.PNG'):
                continue
            path_img = os.path.join(self.img_path, file_name)
            img = cv2.imread(path_img)
            path_lbl = os.path.join(self.img_path, "item_0000" + file_name[-9: -4] + ".xml")
            img = self._ol_xml_2_mask(path_lbl, img)
            video.write(img)
        return video


def main():
    img_path = "/home/xq/文档/projects/video/2021-01-20/task_视频1-2021_01_20_08_09_35-labelme 3.0/default"
    video_path = "/home/xq/文档/projects/video/2021-01-20"
    fps = 24
    img2vdo = Image2Video(img_path, video_path, fps)
    res_video = img2vdo.apply()
    res_video.release()


if __name__ == "__main__":
    main()

