

from __future__ import annotations
from random import shuffle
import itertools
import copy
import argparse
from pycocotools.coco import COCO
import json
import os
import json
from collections import defaultdict
from tqdm import tqdm


def balance(gt_path, gt_balance, each_num):
    coco = COCO(gt_path)
    with open(gt_path, "r") as f:
        data = json.load(f)

    new_imgs = data["images"]
    new_anns = data["annotations"]
    print("before: img_nums = {}, ann_names = {}".format(len(data["images"]), len(data["annotations"])))

    img_balance_dict = defaultdict(list)
    for img_info in data["images"]:
        if "Google" not in img_info["file_name"]:
            continue
        key = img_info["file_name"].split("_")[0]
        img_balance_dict[key].append(img_info)
    print([(x, len(img_balance_dict[x])) for x in img_balance_dict.keys()])

    max_img_id = max([info["id"] for info in data["images"]])
    max_ann_id = max([info["id"] for info in data["annotations"]])
    def repeat_info(add_img_info_list):
        if not add_img_info_list:
            return
        nonlocal max_img_id, max_ann_id
        shuffle(add_img_info_list)
        add_img = itertools.cycle(add_img_info_list)
        for i, img_info in enumerate(add_img):
            if i >= each_num - len(add_img_info_list):
                break
            new_img_info = copy.deepcopy(img_info)
            max_img_id += 1
            new_img_info["id"] = max_img_id
            new_imgs.append(new_img_info)
            # print(img_info["id"], new_img_info["id"], new_img_info)

            ann_ids = coco.getAnnIds(imgIds=[img_info["id"]])
            ann_infos = coco.loadAnns(ann_ids)
            for ann_info in ann_infos:
                new_ann_info = copy.deepcopy(ann_info)
                max_ann_id += 1
                new_ann_info["id"] = max_ann_id
                new_ann_info["image_id"] = max_img_id
                new_anns.append(new_ann_info)
        print(len(add_img_info_list))
        print("after: img_nums = {}, ann_names = {}".format(len(new_imgs), len(new_anns)))
        print(len(set([x["id"] for x in new_imgs])), len(set([x["id"] for x in new_anns])))
    
    repeat_info(img_balance_dict["chair"])
    repeat_info(img_balance_dict["garbage"])
    repeat_info(img_balance_dict["plants"])
    repeat_info(img_balance_dict["half"] + img_balance_dict["office"] + img_balance_dict["sideways"])
    repeat_info(img_balance_dict["ergonomic"])
    # repeat_info(img_balance_dict["ergonomic1"])
    
    print("after: img_nums = {}, ann_names = {}".format(len(new_imgs), len(new_anns)))
    print(len(set([x["id"] for x in new_imgs])), len(set([x["id"] for x in new_anns])))

    shuffle(new_imgs)

    json_dict = dict(
        images=new_imgs,
        annotations=new_anns,
        categories=data["categories"],
        info=data["info"],
        license=data["license"] if "license" in data else [],
    )
    with open(gt_balance, 'w') as f:
        json.dump(json_dict, f, ensure_ascii=False)


#对官方GT json做筛选，只留下需要计算的类别，这些类别存在id_con里。

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, help="")
    parser.add_argument('--gt_balc', type=str, default=None, help="")
    parser.add_argument('--each_num', type=int, help="")

    args = parser.parse_args()

    # train:2500 | val:800
    # val_name = "add_train_person_2022-11-22"
    # each_num = 2500  
    # val_name = "add_val_person_2022-11-22"
    # each_num = 800  
    # gt_path = "/dataset/object_detection/coco_wider_pedestrian/google_images/add_json/{}.json".format(val_name)
    # gt_balc = "/dataset/object_detection/coco_wider_pedestrian/google_images/add_json/{}_balance.json".format(val_name)

    gt_path = args.gt_path
    gt_balc = args.gt_balc if args.gt_balc else os.path.splitext(args.gt_path)[0] + "_balance.json"
    each_num = args.each_num
    balance(gt_path, gt_balc, each_num)