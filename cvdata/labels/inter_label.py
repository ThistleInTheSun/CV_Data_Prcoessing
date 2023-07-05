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
        f.write(json.dumps(data, indent=2, ensure_ascii=False))


def load_bbox(file_dir, idx, group, label):
    path = os.path.join(file_dir, str(idx).rjust(8, "0") + ".json")
    if not os.path.exists(path):
        return None
    data = load_json(path)
    for info in data["shapes"]:
        if "group_id" not in info or info["group_id"] != group or info["label"] != label:
            continue
        tmp = sorted([info["points"][0], info["points"][1]])
        info["bbox"] = tmp[0] + tmp[1]
        info["idx"] = idx
        return info
    return None
    # raise ValueError("no group named", group)


def inter(bbox_begin, bbox_end):
    split_num = (bbox_end['idx'] - bbox_begin['idx'])
    delta_x_min = (bbox_end['bbox'][0] - bbox_begin['bbox'][0]) / split_num
    delta_y_min = (bbox_end['bbox'][1] - bbox_begin['bbox'][1]) / split_num
    delta_x_max = (bbox_end['bbox'][2] - bbox_begin['bbox'][2]) / split_num
    delta_y_max = (bbox_end['bbox'][3] - bbox_begin['bbox'][3]) / split_num
    x_min, y_min, x_max, y_max = bbox_begin['bbox']
    info_list = []
    for idx in range(bbox_begin['idx'] + 1, bbox_end['idx']):
        x_min += delta_x_min
        y_min += delta_y_min
        x_max += delta_x_max
        y_max += delta_y_max
        info = {
            "idx": idx, 
            "bbox": [x_min, y_min, x_max, y_max], 
            "label": bbox_begin["label"],
        }
        if "attribute" in bbox_begin:
            info["attribute"] = bbox_begin["attribute"]
        info_list.append(info)
    return info_list


def write(file_dir, info_list, group, label, attribute=None):
    for info in info_list:
        json_path = os.path.join(file_dir, str(info["idx"]).rjust(8, "0") + ".json")
        if os.path.exists(json_path):
            data = load_json(json_path)
        else:
            data = {
                "imageHeight": 1080,
                "imageWidth": 1920,
                "shapes": []
                }
        
        new_shapes = []
        for i, s in enumerate(data["shapes"]):
            if "group_id" in s and s["group_id"] != group or s["label"] != label:
                new_shapes.append(s)
        data["shapes"] = new_shapes

        x_min, y_min, x_max, y_max = info["bbox"]
        new_info = dict(
            group_id=group,
            label=label,
            points=[[x_min, y_min], [x_max, y_max]],
            shape_type="rectangle",
        )
        if attribute:
            new_info["attribute"] = attribute
        if "attribute" in info:
            new_info["attribute"] = info["attribute"]
        data["shapes"].append(new_info)
        write_json(json_path, data)


def inter_json(file_dir, k_frm, group, label, attribute=None):
    # begin, end = min(begin, end), max(begin, end)
    k_frm = sorted(list(set(k_frm)))
    for i in range(1, len(k_frm)):
        begin, end = k_frm[i - 1], k_frm[i]
        print("inter_json(begin={}, end={}, group={}, label={}, file_dir={})".format(begin, end, group, label, file_dir))
        bbox_begin = load_bbox(file_dir, begin, group, label)
        bbox_end = load_bbox(file_dir, end, group, label)
        bbox_info_list = inter(bbox_begin, bbox_end)
        write(file_dir, bbox_info_list, group, label, attribute)


def set_attribute(file_dir, group=None, begin=-1, end=float("inf"), label=None, attribute=None):
    lis = sorted(os.listdir(file_dir))
    begin = str(begin).rjust(8, "0") + ".json"
    end = str(end).rjust(8, "0") + ".json"
    for json_name in lis:
        if json_name < begin or json_name > end:
            continue
        json_path = os.path.join(file_dir, json_name)
        if os.path.exists(json_path):
            data = load_json(json_path)
        else:
            raise ValueError("no json path:", json_path)

        for info in data["shapes"]:
            if group is not None and info["group_id"] != group:
                continue
            if label is not None and info["label"] != label:
                continue
            if attribute not in ["", None]:
                info["attribute"] = attribute
            elif "attribute" in info:
                del info["attribute"]
        write_json(json_path, data)


def inter_json_by_trk(file_dir, group, label, begin=-1, end=float("inf")):
    print("inter_json_by_trk(begin={}, end={}, group={}, label={}, file_dir={})".format(begin, end, group, label, file_dir))
    info_list = get_all_trk(file_dir, group, label, begin=begin, end=end)
    info_list = inter_by_trk(info_list)
    write(file_dir, info_list, group, label)


def get_all_trk(file_dir, group, label, begin, end):
    lis = sorted(os.listdir(file_dir))
    begin = max(begin, int(lis[0].replace(".json", "")))
    end = min(end, int(lis[-1].replace(".json", "")))
    info_list = []
    for idx in range(begin, end + 1):
        info = load_bbox(file_dir, idx, group, label)
        if info is not None:
            info_list.append(info)
    return info_list


def inter_by_trk(info_list):
    pre_idx = info_list[0]["idx"]
    i = 1
    while i < len(info_list):
        cur_idx = info_list[i]["idx"]
        if cur_idx > pre_idx + 1:
            sub_info_list = inter(info_list[i - 1], info_list[i])
            info_list = info_list[:i] + sub_info_list + info_list[i:]
            i += len(sub_info_list)
        else:
            i += 1
        pre_idx = info_list[i - 1]["idx"]
    return info_list


def change_group(file_dir, label, src_group, tar_group, begin, end):
    print("change_group(src_group={}, tar_group={}, file_dir={})".format(src_group, tar_group, file_dir))
    lis = sorted(os.listdir(file_dir))
    begin = str(begin).rjust(8, "0") + ".json"
    end = str(end).rjust(8, "0") + ".json"
    for json_name in lis:
        if json_name < begin or json_name > end:
            continue
        json_path = os.path.join(file_dir, json_name)
        data = load_json(json_path)
        del_i = None
        chg_i = None
        for i in range(len(data["shapes"])):
            info = data["shapes"][i]
            if info["label"] != label:
                continue
            if "group_id" in info and info["group_id"] == tar_group:
                del_i = i
            if "group_id" in info and info["group_id"] in src_group:
                chg_i = i
        if chg_i is not None:
            data["shapes"][chg_i]["group_id"] = tar_group
            if del_i is not None:
                del data["shapes"][del_i]
        write_json(json_path, data)


def change_attribute(file_dir, src_group=None, tar_group=None):
    print("change_attribute(src_group={}, tar_group={}, file_dir={})".format(src_group, tar_group, file_dir))
    lis = sorted(os.listdir(file_dir))
    for json_name in lis:
        json_path = os.path.join(file_dir, json_name)
        data = load_json(json_path)
        for i in range(len(data["shapes"])):
            info = data["shapes"][i]
            if "attribute" in info and isinstance(info["attribute"], list):
                info["attribute"] = 'chefhat'
        write_json(json_path, data)


def del_trk(file_dir, label, group, begin, end):
    if group == []:
        return
    print("del_trk(group={}, file_dir={})".format(group, file_dir))
    lis = sorted(os.listdir(file_dir))
    begin = str(begin).rjust(8, "0") + ".json"
    end = str(end).rjust(8, "0") + ".json"
    for json_name in lis:
        if json_name < begin or json_name > end:
            continue
        json_path = os.path.join(file_dir, json_name)
        data = load_json(json_path)
        del_i = []
        for i in range(len(data["shapes"])):
            info = data["shapes"][i]
            if info["label"] != label:
                continue
            if "group_id" in info and info["group_id"] in group:
                del_i.append(i)
        for i in del_i[::-1]:
            del data["shapes"][i]
        write_json(json_path, data)


if __name__ == "__main__":
    file_dir = "D:/Users/qing.xiang/projects/show/labelme/L1/test_kaola_ManholeCoverDamaged"
    # group, label = 1, "NonMotorVehicle"
    # k_frm = [0, 1482, 1483,6307]
    # inter_json(file_dir, k_frm, group=group, label=label)

    begin, end = 0,99999999

    src, tar = [155],8
    # change_group(file_dir, label="head", src_group=src, tar_group=tar, begin=begin, end=end)

    # del_trk(file_dir, label="head", group=[0], begin=begin, end=end)

    group, label = 1, "NonMotorVehicle"
    inter_json_by_trk(file_dir, group=group, label=label, begin=begin, end=end)




    # set_attribute(file_dir, group=3, begin=begin, end=end, attribute="")
    
    # for v_name in sorted(os.listdir("D:/Users/qing.xiang/projects/show/labelme/lbls/chefhat")):
    #     if v_name != "chef_mask_hat_18":
    #         continue
    #     path = os.path.join("D:/Users/qing.xiang/projects/show/labelme/lbls/chefhat", v_name)
    #     change_attribute(path)

    # debug_label("/home/sdb1/xq/bmalgo_eval/fire_smoke/fire1_outdor_openroud_light1-1", begin=1641, track_id=1)
