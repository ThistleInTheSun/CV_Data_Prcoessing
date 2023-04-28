import os
import shutil
import base64
import json
from pycocotools.coco import COCO
import argparse
import tqdm
import cv2


def img_encode_b64(img_path):
    with open(img_path, "rb") as bf:
        img_data = bf.read()
    base64_data = base64.b64encode(img_data)
    base64_str = str(base64_data, encoding="utf-8")
    
    return base64_str


def coco2labelme(json_cocofile, imgs_path, save_path, shapeType="rectangle"):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    
    json_coco = COCO(json_cocofile)
    img_data = json_coco.loadImgs(json_coco.getImgIds())

    for img in tqdm.tqdm(img_data):
        img_id = img["id"]
        
        img_name = img["file_name"]
        if os.path.exists(os.path.join(imgs_path, img_name)):
            img_cv = cv2.imread(os.path.join(imgs_path, img_name))
            h, w = img_cv.shape[:2]
        img_width = img.get("width", w)
        img_height = img.get("height", h)
        img_encode_data = img_encode_b64(os.path.join(imgs_path, img_name))

        anno_ids = json_coco.getAnnIds(img_id)
        
        shapes = []
        for anno_id in anno_ids:
            anno = json_coco.loadAnns(anno_id)[0]
            
            cat_id = anno["category_id"]
            cat_label = json_coco.loadCats(cat_id)[0]["name"]
    
            bbox = anno["bbox"]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0]+bbox[2], bbox[1]+bbox[3]
            points = [[x1, y1], [x2, y2]]
            try:
                is_modify = anno["is_modify"]
            except:
                is_modify = 0
            
            shapes.append({"label": cat_label,
                           "is_modify": is_modify,
                           "points": points,
                           "group_id": None,
                           "shape_type": shapeType,
                           "flags": {}})
        
        labelme_data = {"version": "",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": img_name,
                        "imageData": img_encode_data,
                        "imageHeight": img_height,
                        "imageWidth": img_width}
        
        save_json_file = os.path.join(save_path, str(img_name).split(".")[0]+".json")
        if os.path.exists(save_json_file):
            os.remove(save_json_file)

        with open(save_json_file, "w", encoding="utf8") as jp:
            json.dump(labelme_data, jp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_file', default='merge_gt.json', type=str,
                        help="")
    parser.add_argument('-i', '--imgs_path', type=str, default='/home/kelaboss/datasets/coco/images/val2017',
                        help="")
    parser.add_argument('-s', '--save_path', type=str, default='coco2labelme_result',
                        help="")

    arg = parser.parse_args()

    json_file = arg.json_file
    imgs_path = arg.imgs_path
    save_path = arg.save_path

    
    coco2labelme(json_file, imgs_path, save_path)

