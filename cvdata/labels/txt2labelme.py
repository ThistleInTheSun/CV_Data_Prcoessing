import argparse
import json
import os
import sys

import cv2
from tqdm import tqdm

sys.path.append(os.getcwd())
from cvdata.labels.reader import load_L1
from cvdata.labels.saver import JsonSaver


def txt2labelme(anno_dir, save_file):
    saver = JsonSaver(save_file)
    for txt_name in tqdm(sorted(os.listdir(anno_dir))):
        path = os.path.join(anno_dir, txt_name)
        data = load_L1(path, 1280, 720)
        saver(res=data, name=txt_name.replace(".txt", ".json"))


if __name__ == "__main__":
    txt_dir = "/home/sdb1/video_event/L1/57_h264/57_h264_anno"
    save_file = "/home/sdb1/video_event/L1/57_h264/57_h264"
    txt2labelme(txt_dir, save_file)
