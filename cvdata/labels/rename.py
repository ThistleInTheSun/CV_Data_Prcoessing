from __future__ import annotations

import os
from tqdm import tqdm
from shutil import copy
import argparse


def rename(src_dir, trg_dir, str_name):
    print(src_dir, trg_dir, str_name)
    os.makedirs(trg_dir, exist_ok=True)
    img_names = [x for x in os.listdir(src_dir) if not x.endswith(".json")]

    for img_n in tqdm(img_names):
        name = os.path.splitext(img_n)[0]
        src_img_path = os.path.join(src_dir, img_n)
        tar_img_path = os.path.join(trg_dir, str_name + "_" + img_n)
        copy(src_img_path, tar_img_path)
        if os.path.exists(os.path.join(src_dir, name + ".json")):
            src_path = os.path.join(src_dir, name + ".json")
            tar_path = os.path.join(trg_dir, str_name + "_" + name + ".json")
            copy(src_path, tar_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_dir', type=str, help="")
    parser.add_argument('-t', '--trg_dir', type=str, help="")
    parser.add_argument('-n', '--name', type=str, help="")

    args = parser.parse_args()

    rename(args.src_dir, args.trg_dir, args.name)
