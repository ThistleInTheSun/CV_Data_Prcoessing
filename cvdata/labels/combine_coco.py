

from __future__ import annotations
from random import shuffle
import argparse
import json
import os
import json
from collections import defaultdict
import os
from tqdm import tqdm


def combine_coco(gt_list, save_path):
    data_list = []
    print("reading ...")
    for gt_path in gt_list:
        print(gt_path)
        if not os.path.exists(gt_path):
            raise ValueError("not exist {}".format(gt_path))
        with open(gt_path, "r") as f:
            data = json.load(f)
        data_list.append(data)
    img_ids = {}  # img_id: img_info
    categories_ids = {}
    anno_ids = {}
    print("processing ...")
    for data in data_list:
        dic_img_id = {}  # old: new
        print("sub_data: img_nums = {}, ann_names = {}".format(len(data["images"]), len(data["categories"])))
        for img_info in tqdm(data["images"], desc="images"):
            cur_img_id = img_info["id"]
            if cur_img_id not in img_ids:
                img_ids[cur_img_id] = img_info
            else:
                if img_info["file_name"] == img_ids[cur_img_id]["file_name"]:
                    continue
                else:
                    new_id = max(img_ids.keys()) + 1
                    dic_img_id[cur_img_id] = new_id
                    img_info["id"] = new_id
                    img_ids[new_id] = img_info

        dic_categories_id = {}  # old: new
        # print("categories:")
        for categories_info in tqdm(data["categories"], desc="categories"):
            cur_categories_id = int(categories_info["id"])
            if cur_categories_id in categories_ids:
                if categories_info["name"] == categories_ids[cur_categories_id]["name"]:
                    continue
                else:
                    new_id = max(categories_ids.keys()) + 1
                    dic_categories_id[cur_categories_id] = new_id
                    categories_info["id"] = new_id
                    categories_ids[new_id] = categories_info
            else:
                categories_ids[cur_categories_id] = categories_info
        
        dic_anno_id = {}  # old: new
        # print("annotations:")
        for anno_info in tqdm(data["annotations"], desc="annotations"):
            if anno_info["image_id"] in dic_img_id:
                anno_info["image_id"] = dic_img_id[anno_info["image_id"]]
            if anno_info["category_id"] in dic_categories_id:
                anno_info["category_id"] = dic_categories_id[anno_info["category_id"]]
            cur_anno_id = anno_info["id"]
            if cur_anno_id in anno_ids:
                new_id = max(anno_ids.keys()) + 1
                dic_anno_id[cur_anno_id] = new_id
                anno_info["id"] = new_id
                anno_ids[new_id] = anno_info
            else:
                anno_ids[cur_anno_id] = anno_info
        print("after: img_nums = {}, ann_names = {}".format(len(img_ids.values()), len(anno_ids.values())))
        print(len(set([x["id"] for x in img_ids.values()])), len(set([x["id"] for x in anno_ids.values()])))

    json_dict = dict(
        images=list(img_ids.values()),
        annotations=list(anno_ids.values()),
        categories=list(categories_ids.values()),
        info="",
        license=[],
    )
    with open(save_path, 'w') as f:
        json.dump(json_dict, f, ensure_ascii=False)


#对官方GT json做筛选，只留下需要计算的类别，这些类别存在id_con里。

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco1', type=str, default="", help="")
    parser.add_argument('--coco2', type=str, default="", help="")
    parser.add_argument('--coco_list', type=str, nargs='+', default="", help="")
    parser.add_argument('--res_coco', type=str, help="")
    args = parser.parse_args()

    if args.coco1 and args.coco2:
        combine_coco([args.coco1, args.coco2], args.res_coco)
    elif args.coco_list:
        combine_coco(args.coco_list, args.res_coco)
    else:
        ValueError("request (both coco1 and coco2) or coco_list")