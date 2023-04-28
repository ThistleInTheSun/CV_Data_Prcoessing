from __future__ import annotations
import argparse
import json
import os
import json
from collections import defaultdict
import os
from warnings import warn


def get_img_nums(gt_dir):
    if os.path.isdir(gt_dir):
        files = sorted(os.listdir(gt_dir))
    elif gt_dir.endswith(".json"):
        files = [gt_dir]
    else:
        raise ValueError("it is not a dir or a json.")

    for f in files:
        if not f.endswith(".json"):
            continue
        gt_path = os.path.join(gt_dir, f)

        with open(gt_path, "r") as f:
            data = json.load(f)
        ad = 0
        sur = 0
        coco = 0
        other = 0
        ad_ids = set()
        sur_ids = set()
        coco_ids = set()
        other_ids = set()
        anno_ids = set()
        for info in data["images"]:
            if info["file_name"].startswith("ad"):
                ad += 1
                ad_ids.add(info["id"])
            elif info["file_name"].startswith("sur"):
                sur += 1
                sur_ids.add(info["id"])
            elif info["file_name"].startswith("0000"):
                coco += 1
                coco_ids.add(info["id"])
            else:
                other += 1
                other_ids.add(info["id"])
                # print(info["file_name"])
        cls = defaultdict(int)
        for info in data["annotations"]:
            cls[info["category_id"]] += 1
            anno_ids.add(info["id"])
        anno = len(data["annotations"])
        categ = len(data["categories"])
        categ_ids = set([x["id"] for x in data["categories"]])
        print("\n{}".format(f))
        print("ad: {}, sur: {}, coco: {}, other: {}".format(ad, sur, coco, other))

        def namestr(obj, namespace):
            return [name for name in namespace if namespace[name] is obj]

        for n, sn in [(ad, ad_ids), (sur, sur_ids), (coco, coco_ids), (other, other_ids), \
                      (anno, anno_ids), (categ, categ_ids)]:
            if n != len(sn):
                # warn("{} != {}".format(n, len(sn)))
                print('\033[1;30;47m {} != {} \033[0m'.format(n, len(sn)))
        print("ad: {}, sur: {}, coco: {}, other: {}".format(len(ad_ids), len(sur_ids), len(coco_ids), len(other_ids)))
        print(cls)
        print(data["categories"])


if __name__ == "__main__":
    # gt_path = "/home/sdb1/xq/taiwan/data/object_detection/coco_wider_pedestrian/google_images/add_json"
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="")
    args = parser.parse_args()

    get_img_nums(args.path)
