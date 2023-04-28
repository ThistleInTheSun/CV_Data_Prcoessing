

from __future__ import annotations
from random import shuffle
from shutil import copyfile
import json
import os
import json
import os
from tqdm import tqdm
import argparse


def coco_get_sub(num, gt_path, gt_sub, txt_sub, img_sub_dir, img_root):
    with open(gt_path, "r") as f:
        data = json.load(f)
    shuffle(data["images"])
    new_images = data["images"][:num]
    new_image_ids = [info["id"] for info in new_images]

    # new gt anno
    annos = data["annotations"]
    new_annos = []
    for info in tqdm(annos):
        if info["image_id"] in new_image_ids:
            new_annos.append(info)
    json_dict = dict(
        images=new_images,
        annotations=new_annos,
        categories=data["categories"],
        info=data["info"],
        license=data["license"] if "license" in data else [],
    )
    with open(gt_sub, 'w') as f:
        json.dump(json_dict, f, ensure_ascii=False)

    # new image name txt
    new_image_names = sorted([info["file_name"] for info in new_images])
    with open(txt_sub, 'w') as f:
        for img_n in new_image_names:
            f.write(img_n + "\n")

    # new images dir
    if img_sub_dir:
        os.makedirs(img_sub_dir, exist_ok=True)
        for img_info in new_images:
            org_path = os.path.join(img_root, img_info["file_name"])
            tar_path = os.path.join(img_sub_dir, img_info["file_name"])
            copyfile(org_path, tar_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, help="")
    parser.add_argument('--num', type=int, default=100, help="")
    parser.add_argument('--gt_sub', type=str, default=None, help="")
    parser.add_argument('--txt_sub', type=str, default=None, help="")
    parser.add_argument('--img_root', type=str, default=None, help="")
    parser.add_argument('--img_sub_dir', type=str, default=None, help="")
    args = parser.parse_args()

    if not args.gt_sub:
        args.gt_sub = os.path.splitext(args.gt_path)[0] + "_{}.json".format(args.num)
    if not args.txt_sub:
        args.txt_sub = os.path.splitext(args.gt_path)[0] + "_{}.txt".format(args.num)
    if not args.img_root:
        args.img_sub_dir = None
    elif not args.img_sub_dir:
        args.img_sub_dir = "images_" + os.path.splitext(args.gt_path)[0] + "_{}".format(args.num)
    coco_get_sub(args.num, args.gt_path, args.gt_sub, args.txt_sub, args.img_sub_dir, args.img_root)
