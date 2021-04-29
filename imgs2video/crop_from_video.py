# -*- coding: UTF-8 -*-
import os
from typing import *

import cv2
from tqdm import tqdm
from manuvision_img_utils.core.io.reader import Reader
from manuvision_img_utils.core.io.writer import Writer

__all__ = ["GenVideo"]


class GenVideo(object):
    def __init__(self,
                 input_path: Text,
                 img_size: Tuple[int, int],
                 video_fps: int,
                 output_path: Text = None,
                 transform_model=None,
                 ):
        self.input_path = input_path
        self.img_size = img_size
        self.video_fps = video_fps
        self.text_color = (0, 0, 255)
        self.location = (img_size[0] - 150, 50)
        self.transform_model = transform_model

        if os.path.isdir(input_path):
            self.imgR = Reader(input_path)
            # self.output_path = output_path if output_path else input_path + "_output.mp4"
        else:
            self.imgR = VideoReader(input_path)
            self.output_path = output_path if output_path else os.path.splitext(input_path)[0] + "_output.mp4"

    def apply(self, save_img_path):
        writer = Writer(save_img_path)
        i = 0
        try:
            for fn in tqdm(self.imgR.filenames):
                img = self.imgR.get_image(fn)
                if img is None:
                    print("img is None")
                    break
                res_img = self.transform_model.draw_infer(img)
                # name = str(i).rjust(6, "0") + ".jpg"
                # name = "ch08_20210328130500_" + str(i).rjust(6, "0") + ".jpg"
                fn = os.path.splitext(fn)[0] + "_cropped"
                name = "{}.jpg".format(fn)
                # print(name)
                writer.save(res_img, name)
                i += 1
        finally:
            self.imgR.cap.release()
        return


class Crop(object):
    def __init__(self, roi=None):
        self.roi = roi

    def draw_infer(self, img):
        sml_img = img[self.roi[1]: self.roi[3], self.roi[0]: self.roi[2]]
        return sml_img


def detect_video_and_save_img(path, save_img_path):
    size = (640, 368)
    fps = 1
    # transform_model = Crop([700, 570, 1050, 860])
    transform_model = Crop([700 // 2, 570 // 2, 1050 // 2, 860 // 2])
    # transform_model = Crop([640, 580, 930, 810])
    img2vdo = GenVideo(path, size, fps, transform_model=transform_model)
    img2vdo.apply(save_img_path)


def composite_video():
    from tqdm import tqdm

    input_path = "/home/xq/文档/projects/加油站/video_img"
    save_path = "video" + ".mp4"
    fps = 24
    img_size = (1920, 1080)

    imgR = Reader(input_path)
    imgR.filenames = sorted(imgR.filenames)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(save_path, fourcc, fps, img_size)

    for fn in tqdm(imgR.filenames):
        img = imgR.get_image(fn)
        cv2.imshow("img", img)
        cv2.waitKey()
        video.write(img)
    video.release()
    return video


if __name__ == "__main__":
    import sys

    curr_path = sys.path[0]
    path = "/home/xq/文档/projects/加油站/data/地堆/0405"
    save_img_path = "/home/xq/文档/projects/加油站/data/地堆/0405_cropped"
    detect_video_and_save_img(path, save_img_path)

    # composite_video()
