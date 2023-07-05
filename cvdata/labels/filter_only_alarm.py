import json
import os
import sys
from collections import defaultdict

from tqdm import tqdm

sys.path.append(os.getcwd())
from cvdata.labels.reader import load_labelme
from cvdata.labels.saver import JsonSaver


def filter_glove(file_dir, save_file):
    lis = sorted(os.listdir(file_dir))
    saver = JsonSaver(save_file)
    for json_name in lis:
        json_path = os.path.join(file_dir, json_name)
        data, img_info = load_labelme(json_path)
        new_data = []
        for info in data:
            if info["label"] != "hand":
                continue
            if "attribute" in info and info["attribute"] != "noGlove":
                continue
            new_data.append(info)
        saver(res=new_data, name=json_name, 
              w=img_info.get("imageWidth"), h=img_info.get("imageHeight"))


if __name__ == "__main__":
    file_dir = "D:/Users/qing.xiang/projects/show/labelme/lbls/glove/glove"
    save_dir = "D:/Users/qing.xiang/projects/show/labelme/lbls/glove/glove_filter"
    
    for v_name in sorted(os.listdir(file_dir)):
        # if "_9" not in v_name:
        #     continue
        print(v_name)
        dir_path = os.path.join(file_dir, v_name)
        save_path = os.path.join(save_dir, v_name)
        filter_glove(dir_path, save_path)
