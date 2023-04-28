import argparse
import json
import os
import sys

import cv2
from tqdm import tqdm

sys.path.append(os.getcwd())
from cvdata.labels.reader import load_labelme


class CocoSaver():
    def __init__(self, path, classes_dict):
        self.anno_id = 0

        self.cat_name_id = {}
        if isinstance(classes_dict, dict):
            self.cat_name_id = classes_dict
        elif isinstance(classes_dict, str):
            with open(classes_file, "r", encoding="utf8") as tf:
                classes_dict = json.load(tf)
            for class_name, label in classes_dict.items():
                self.cat_name_id[class_name] = int(label)

        self.cat_list = []
        for class_name, label in self.cat_name_id.items():
            self.cat_list.append({
                "id": int(label),
                "name": class_name,
                "supercategory": "",
            })
        
    
    def gen_coco_info(self, res, img_id, name, img=None, is_modify=None, *args, **kwargs):
        annos_list = []
        for info in res:
            x1, y1, x2, y2 = info["bbox"]
            coco_x, coco_y = min(x1, x2), min(y1, y2)
            coco_w, coco_h = max(x1, x2) - min(x1, x2), max(y1, y2) - min(y1, y2)
            coco_area = coco_w * coco_h
            cat_id = self.cat_name_id[info["label"]]
            coco_info = {
                "id": self.anno_id,
                "image_id": img_id,
                "category_id": cat_id,
                "is_modify": is_modify,
                "segmentation": [x1, y1, x2, y1, x2, y2, x1, y2],
                "area": coco_area,
                "bbox": [coco_x, coco_y, coco_w, coco_h],
                "iscrowd": 0,
            }
            if "score" in info:
                coco_info["score"] = info["score"]
            self.anno_id += 1
            annos_list.append(coco_info)
        return annos_list


def labelme2coco(labelme_dir, classes_dict, save_file, img_dir=None, gt_json=None, start_list=None):
    

    saver = CocoSaver(save_file, classes_dict)
    # cat_list = []
    # cat_name_id = {}
    # for class_name, label in classes_dict.items():
    #     cla_id = int(label)
    #     cat_name_id[class_name] = cla_id
    #     cat_list.append({"id": cla_id,
    #                       "name": class_name,
    #                       "supercategory": ""})

    imgs_list = []
    annos_list = []
    print("gt_json:", gt_json)
    print("img_dir:", img_dir)
    if gt_json:
        imgs_list = json.load(open(gt_json))["images"]
        for gt_info in tqdm(imgs_list):
            img_id = gt_info["id"]
            file_name = gt_info["file_name"]
            json_path = os.path.join(labelme_dir, os.path.splitext(file_name)[0] + ".json")
            if not os.path.exists(json_path):
                random_lbl = list(saver.cat_name_id.keys())[0]
                data = [{"bbox": [0, 0, 1, 1], "score": 0.01, "label": random_lbl}]
                coco_anno_info = saver.gen_coco_info(data, img_id, file_name)
                annos_list += coco_anno_info
                continue
            data = load_labelme(json_path)
            coco_anno_info = saver.gen_coco_info(data, img_id, file_name)
            annos_list += coco_anno_info
    elif img_dir:
        print("gen gt coco")
        img_name_list = sorted([x for x in os.listdir(img_dir) if os.path.splitext(x)[-1] in [".jpeg", ".jpg", ".png"]])
        for img_id, file_name in enumerate(tqdm(img_name_list)):
            name = os.path.splitext(file_name)[0]
            img = cv2.imread(os.path.join(img_dir, file_name))
            h, w = img.shape[:2]
            json_path = os.path.join(labelme_dir, name + ".json")
            imgs_list.append({"id": img_id,
                        "width": w,
                        "height": h,
                        "file_name": file_name})
            if os.path.exists(json_path):
                data = load_labelme(json_path)
                coco_anno_info = saver.gen_coco_info(data, img_id, file_name)
                annos_list += coco_anno_info
    else:
        json_name_list = sorted([x for x in os.listdir(img_dir) if os.path.splitext(x)[-1] == ".json"])
        with open(labelme_path, "r", encoding="utf8") as jf:
            labelmedata = json.load(jf)

        imgs_list.append({"id": img_id,
                          "width": w,
                          "height": h,
                          "file_name": img_name, })

        labelmeshapes = labelmedata["shapes"]
        for shape in labelmeshapes:
            is_modify = shape.get("is_modify", None)  # 是否 通过 labelme 修改 确定为 gt 
            
            cat_label = shape["label"]  # category 的名称
            cat_id = cat_name_id[cat_label]  # category 的编号

            x1, y1, x2, y2 = shape["points"][0][0], shape["points"][0][1], shape["points"][1][0], shape["points"][1][1]
            coco_x, coco_y = min(x1, x2), min(y1, y2)   # 目标框左上角坐标
            coco_w, coco_h = max(x1, x2)-min(x1, x2), max(y1, y2)-min(y1, y2)   # 目标框宽、高
            coco_area = coco_w*coco_h   # 目标框面积

            annos_list.append({"id": anno_id,
                               "image_id": img_id,
                               "category_id": cat_id,
                               "is_modify": is_modify,
                               "segmentation": [x1, y1, x2, y1, x2, y2, x1, y2],
                               "area": coco_area,
                               "bbox": [coco_x, coco_y, coco_w, coco_h],
                               "iscrowd": 0})
            if "score" in shape:
                annos_list[-1]["score"] = shape["score"]
            anno_id += 1
    
    coco_file = {"info": {},
                 "license": [],
                 "images": imgs_list,
                 "annotations": annos_list,
                 "categories": saver.cat_list}

    if os.path.exists(save_file):
        os.remove(save_file)
    with open(save_file, "w", encoding="utf8") as fp:
        json.dump(coco_file, fp)


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

    # labelme_path = "/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu_labelme"
    # save_file = "/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu.json"
    # img_dir = "/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu_img"
    # gt_json = None

    labelme_path = "/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu_det"
    save_file = "/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu_det.json"
    img_dir = "/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu_img"
    gt_json = "/home/sdb1/video_event/face_mask/imgs_kitchen_yuntu/imgs_kitchen_yuntu.json"

    classes_file = {"head": 1}
    labelme2coco(labelme_path, classes_file, save_file, img_dir, gt_json)
