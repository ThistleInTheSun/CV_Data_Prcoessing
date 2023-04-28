from __future__ import annotations

import os
from tqdm import tqdm
from random import shuffle
from shutil import copy
import argparse


def split_trainval(path, train_path, val_path, img_type, rate, is_shuffle):
    print(path)
    img_names = [x for x in os.listdir(path) if not x.endswith(".json")]
    if is_shuffle:
        shuffle(img_names)
    thr = int(len(img_names) * rate)
    # train
    for img_n in tqdm(img_names[:thr], desc="train"):
        name = os.path.splitext(img_n)[0]
        src_img_path = os.path.join(path, img_n)
        tar_img_path = os.path.join(train_path, img_type + "_" + img_n)
        copy(src_img_path, tar_img_path)
        if os.path.exists(os.path.join(path, name + ".json")):
            src_path = os.path.join(path, name + ".json")
            tar_path = os.path.join(train_path, img_type + "_" + name + ".json")
            copy(src_path, tar_path)
    # val
    for img_n in tqdm(img_names[thr:], desc="val"):
        name = os.path.splitext(img_n)[0]
        src_img_path = os.path.join(path, img_n)
        tar_img_path = os.path.join(val_path, img_type + "_" + img_n)
        copy(src_img_path, tar_img_path)
        if os.path.exists(os.path.join(path, name + ".json")):
            src_path = os.path.join(path, name + ".json")
            tar_path = os.path.join(val_path, img_type + "_" + name + ".json")
            copy(src_path, tar_path)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_ann_dir', type=str, help="")
    parser.add_argument('--img_type', type=str, help="")
    parser.add_argument('--train_path', type=str, help="")
    parser.add_argument('--val_path', type=str, help="")
    parser.add_argument('--rate', type=float, default=0.667, help="")
    parser.add_argument('--is_shuffle', type=bool, default=True, help="")
    args = parser.parse_args()

    img_ann_dir = args.img_ann_dir
    img_type = args.img_type
    train_path = args.train_path
    val_path = args.val_path
    rate = args.rate
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    split_trainval(img_ann_dir, train_path, val_path, img_type, rate, args.is_shuffle)