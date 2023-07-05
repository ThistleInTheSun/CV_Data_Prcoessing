import argparse
import json
import os
import shutil
import sys
from random import shuffle

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
from cvdata.labels.saver import ImgSaver


def video2imgs(video_path, save_dir):
    saver = ImgSaver(save_dir, is_draw=False)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while True:
        if frame_id % 20 == 0:
            print('Processing frame {}'.format(frame_id))
        ret_val, frame = cap.read()
        if not ret_val:
            break
        img = frame
        name = str(frame_id).rjust(8, "0")
        saver([], name=name, img=img)
        frame_id += 1


def labelme2yolo_allvideos(labelme_dir, classes_dict, save_file, img_dir=None, start_list=None):
    pass

if __name__ == "__main__":
    video_path = "D:/Users/qing.xiang/projects/show/labelme/shirtless_YouTube.mp4"
    save_dir = "D:/Users/qing.xiang/projects/show/labelme/imgs/shirtless_YouTube"
    video2imgs(video_path, save_dir)
