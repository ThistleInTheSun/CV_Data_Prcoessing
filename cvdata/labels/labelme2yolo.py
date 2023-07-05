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
from cvdata.labels.reader import load_json, load_labelme

train_val_rate = 0.8


def json2txt(json_path, save_txt_path, classes_dict):
    if not os.path.exists(json_path):
        return
    data = load_json(json_path)
    img_w, img_h = data['imageWidth'], data['imageHeight']
    labels = []
    for info in data["shapes"]:
        pts = info['points']
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x = (x1 + x2) / 2 / img_w 
        y = (y1 + y2) / 2 / img_h
        w  = abs(x2 - x1) / img_w
        h  = abs(y2 - y1) / img_h
        cid = classes_dict[info['label']]        
        labels.append([cid, x, y, w, h])
    if len(labels) > 0:
        np.savetxt(save_txt_path, labels)


def labelme2yolo(labelme_dir_path, img_dir_path, save_file, classes_dict, v_name=""):
    imgs_lis = sorted(os.listdir(img_dir_path))[0::10]
    shuffle(imgs_lis)
    n = int(train_val_rate * len(imgs_lis))
    train_lis, val_lis = imgs_lis[:n], imgs_lis[n:]
    images_train = os.path.join(save_file, "images", "train")
    images_val = os.path.join(save_file, "images", "val")
    labels_train = os.path.join(save_file, "labels", "train")
    labels_val = os.path.join(save_file, "labels", "val")
    os.makedirs(images_train, exist_ok=True)
    os.makedirs(images_val, exist_ok=True)
    os.makedirs(labels_train, exist_ok=True)
    os.makedirs(labels_val, exist_ok=True)
    for img_name in tqdm(train_lis):
        img_path = os.path.join(img_dir_path, img_name)
        json_path = os.path.join(labelme_dir_path, img_name.replace(".jpg", ".json"))
        save_img_name = v_name + "_" + img_name
        save_img_path = os.path.join(images_train, save_img_name)
        save_txt_path = os.path.join(labels_train, save_img_name.replace(".jpg", ".txt"))
        shutil.copy(img_path, save_img_path)
        json2txt(json_path, save_txt_path, classes_dict)
    for img_name in val_lis:
        img_path = os.path.join(img_dir_path, img_name)
        json_path = os.path.join(labelme_dir_path, img_name.replace(".jpg", ".json"))
        save_img_name = v_name + "_" + img_name
        save_img_path = os.path.join(images_val, save_img_name)
        save_txt_path = os.path.join(labels_val, save_img_name.replace(".jpg", ".txt"))
        shutil.copy(img_path, save_img_path)
        json2txt(json_path, save_txt_path, classes_dict)


def labelme2yolo_allvideos(labelme_dir, classes_dict, save_file, img_dir=None, start_list=None):
    v_name_labelme_lis = os.listdir(labelme_dir)
    v_name_img_lis = os.listdir(img_dir)
    v_name_lis = set(v_name_labelme_lis).union(set(v_name_img_lis))
    v_name_lis = sorted(v_name_lis)
    for v_name in v_name_lis:
        if v_name != "chef_mask_hat_17":
            continue
        print("v_name:", v_name)
        labelme_dir_path = os.path.join(labelme_dir, v_name)
        img_dir_path = os.path.join(img_dir, v_name)
        if not os.path.isdir(labelme_dir_path) or not os.path.isdir(img_dir_path):
            continue
        labelme2yolo(labelme_dir_path, img_dir_path, save_file, classes_dict, v_name=v_name)


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

    '''
    classes_file.json: {"person":1}
    '''

    labelme_path = "/home/sdb1/xq/bmalgo_eval/chefhat"
    save_file = "/home/sdb1/xq/algorithm/yolov8_train/datasets/chefhead_det"
    img_dir = "/home/sdb1/eco_algo_q2/eco_algo_image"

    classes_file = {"head": 0}
    labelme2yolo_allvideos(labelme_path, classes_file, save_file, img_dir)
