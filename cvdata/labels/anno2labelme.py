import argparse
import json
import os
import sys

import cv2
from tqdm import tqdm

sys.path.append(os.getcwd())
from cvdata.labels.reader import load_anno
from cvdata.labels.saver import JsonSaver


def anno2labelme(anno_dir, save_file):
    saver = JsonSaver(save_file)
    for json_name in tqdm(sorted(os.listdir(anno_dir))):
        path = os.path.join(anno_dir, json_name)
        data = load_anno(path)
        saver(res=data, name=json_name)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-l', '--labelme_path', type=str, default='yolo2labelme_result', help="")
    # parser.add_argument('-c', '--classes_file', type=str, default='classes_self.json', help="")
    # parser.add_argument('-s', '--save_file', type=str, default='labelme2coco_result.json', help="")
    # parser.add_argument('--start_list', type=str, nargs='+', default="", help="")

    # args = parser.parse_args()

    # labelme_path = args.labelme_path
    # classes_file = args.classes_file
    # save_file = args.save_file

    anno_dir = "/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu_anno"
    save_file = "/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu_labelme"

    anno2labelme(anno_dir, save_file)
