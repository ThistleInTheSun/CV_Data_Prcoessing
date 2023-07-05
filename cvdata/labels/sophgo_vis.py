"""
visual gt (format is yolo txt) and pr (format is cvimodel txt)
"""

import cv2
import os
from tqdm import tqdm
import sys

sys.path.append(os.getcwd())
from cvdata.labels.reader import load_yolo_txt, load_cvires_txt
from cvdata.labels.visualize import plot_tracking


if __name__ == "__main__":
    img_dir = "/dataset/head_person_train/val2017_2/images"
    gt_dir = "/dataset/head_person_train/val2017_2/labels"
    pr_dir = "/dataset/head_person_train/val2017_2/predict/labels"
    save_dir = "/dataset/head_person_train/val2017_2/vis"
    os.makedirs(save_dir, exist_ok=True)
    img_list = os.listdir(img_dir)
    gt_empty, pr_empty = [], []
    for img_name in tqdm(sorted(img_list)):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        gt_path = os.path.join(gt_dir, img_name.replace(".jpg", ".txt"))
        gt_info = load_yolo_txt(gt_path, img_w, img_h)
        pr_path = os.path.join(pr_dir, img_name.replace(".jpg", ".txt"))
        # pr_info = load_cvires_txt(pr_path)
        pr_info = load_yolo_txt(pr_path, img_w, img_h)
        if not gt_info and not pr_info:
            continue
        if not gt_info:
            gt_empty.append(img_name)
        if not pr_info:
            pr_empty.append(img_name)
            continue
        img = plot_tracking(img, gt_info, color=(0, 255, 0), line_thickness=2, text_thickness=2)
        img = plot_tracking(img, pr_info, color=(0, 0, 255), line_thickness=1, text_thickness=1)
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img)
    print("gt_empth num: {}, list: {}".format(len(gt_empty), gt_empty))
    print("pr_empty num: {}, list: {}".format(len(pr_empty), pr_empty))