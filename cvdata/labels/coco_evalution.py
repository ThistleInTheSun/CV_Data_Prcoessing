

import json
import os
import time

import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == "__main__":

    gt_path = '/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu.json'
    save_json = '/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu_det.json'

    cocoGt = COCO(gt_path)
    anns = json.load(open(save_json))

    if isinstance(anns, dict):
        cocoDt = cocoGt.loadRes(anns["annotations"])
    elif isinstance(anns, list):
        cocoDt = cocoGt.loadRes(anns)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")

    # cocoEval.params.catIds = [id]
    # print("Id = {}".format(id))
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    iou = 0.5
    num = 0
    for ann in cocoDt.anns.values():
        if ann["score"] >= 0.5:
            num += 1
    print(len(cocoGt.anns), num)


