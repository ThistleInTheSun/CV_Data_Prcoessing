import os
import json
from tqdm import tqdm
from collections import defaultdict



def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        raise ValueError("{} cannot open".format(path))
    return data


def write_json(path, data):
    with open(os.path.join(path), "w") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def change_attribute(file_dir):
    lis = sorted(os.listdir(file_dir))
    for json_name in lis:
        json_path = os.path.join(file_dir, json_name)
        data = load_json(json_path)
        for i in range(len(data["shapes"])):
            info = data["shapes"][i]
            # if "attribute" in info and (info["attribute"] == "shirtless" or isinstance(info["attribute"], list)):
            info["attribute"] = 'nakedBody'
        write_json(json_path, data)


# if __name__ == "__main__":
#     file_dir = "D:/Users/qing.xiang/projects/show/labelme/lbls/shirtless"
#     # file_dir = "/home/sdb1/xq/bmalgo_eval/shirtless"
    
#     for v_name in sorted(os.listdir(file_dir)):
#         if "_9" not in v_name:
#             continue
#         print(v_name)
#         dir_path = os.path.join(file_dir, v_name)
#         change_attribute(dir_path)
    

# def delate_empty_json(file_dir):
#     import shutil
#     for json_name in os.listdir(file_dir):
#         json_path = os.path.join(file_dir, json_name)
#         data = load_json(json_path)
#         # set w h
#         data["imageHeight"] = 1080
#         data["imageWidth"] = 1920
#         # del empty attribute
#         for info in data["shapes"]:
#             if "attribute" in info and info["attribute"] in ["", None]:
#                 del info["attribute"]
#         write_json(json_path, data)  
#         # del empty json
#         if data["shapes"] == []:
#             print(json_name)
#             shutil.move(json_path, os.path.join(file_dir, "empty_" + json_name))

# if __name__ == "__main__":
#     # file_dir = "D:/Users/qing.xiang/projects/show/labelme/lbls/chefhat/chef_mask_hat_2"
#     file_dir = "/home/sdb1/xq/bmalgo_eval/chefhat/chef_hat_1"
#     print(file_dir)
#     delate_empty_json(file_dir)


def add_w_h(file_dir):
    lis = sorted(os.listdir(file_dir))
    for json_name in lis:
        json_path = os.path.join(file_dir, json_name)
        data = load_json(json_path)
        data["imageWidth"] = 1280
        data["imageHeight"] = 720
        write_json(json_path, data)

if __name__ == "__main__":
    file_dir = "D:/Users/qing.xiang/projects/show/labelme/L1/57_h264"
    add_w_h(file_dir)


# def name_begin_0(img_dir, gt_dir):
#     import shutil
#     img_list = sorted(os.listdir(img_dir))
#     begin = int(img_list[0].replace(".jpg", ""))
#     for img_name in sorted(os.listdir(img_dir)):
#         frm_id = int(img_name.replace(".jpg", ""))
#         src = os.path.join(img_dir, img_name)
#         tar = os.path.join(img_dir, str(frm_id - begin).rjust(8, "0") + ".jpg")
#         # print(src, tar)
#         shutil.move(src, tar)
#         json_name = img_name.replace(".jpg", ".json")
#         if not os.path.exists(os.path.join(gt_dir, json_name)):
#             continue
#         src = os.path.join(gt_dir, json_name)
#         tar = os.path.join(gt_dir, str(frm_id - begin).rjust(8, "0") + ".json")
#         # print(src, tar)
#         shutil.move(src, tar)

# if __name__ == "__main__":
#     img_root = "D:/Users/qing.xiang/projects/show/labelme/imgs"
#     file_dir = "D:/Users/qing.xiang/projects/show/labelme/lbls/shirtless"
#     for v_name in sorted(os.listdir(file_dir)):
#         if "shirtless_YouTube_12" not in v_name:
#             continue
#         print(v_name)
#         name_begin_0(os.path.join(img_root, v_name), os.path.join(file_dir, v_name))