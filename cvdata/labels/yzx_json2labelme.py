import json
import os
import sys
from collections import defaultdict

from tqdm import tqdm

sys.path.append(os.getcwd())
from cvdata.labels.reader import load_json
from cvdata.labels.saver import JsonSaver


def json2labelme(file_path, save_file):
    data = load_json(file_path)
    saver = JsonSaver(save_file)
    for json_name in data:
        new_data = []
        for line in data[json_name]:
            # if info["label"] != "hand":
            #     continue
            # if "attribute" in info and info["attribute"] != "noGlove":
            #     continue
            x_min, y_min, x_max, y_max, score, label = line
            info = dict(
                bbox=[int(x) for x in [x_min, y_min, x_max, y_max]],
                label=int(label),
                score=score,
            )
            new_data.append(info)
        saver(res=new_data, name=json_name)


if __name__ == "__main__":
    file_dir = "D:/Users/qing.xiang/projects/show/labelme/L1/results"
    save_dir = "D:/Users/qing.xiang/projects/show/labelme/L1/results_labelme"
    
    for v_name in sorted(os.listdir(file_dir)):
        # if "_9" not in v_name:
        #     continue
        print(v_name)
        file_path = os.path.join(file_dir, v_name)
        save_path = os.path.join(save_dir, v_name)
        json2labelme(file_path, save_path)
