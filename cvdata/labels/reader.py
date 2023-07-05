import json
import os
from warnings import warn


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except:
            print(file_path)
            raise e
    return data


def load_txt(file_path):
    lines = []
    with open(file_path) as f:
        for line in f:
            line = line.replace("\n", "")
            lines.append(line)
    return lines


def load_labelme(path):
    if not os.path.exists(path):
        raise ValueError("not exist", path)
    data = load_json(path)
    info_list = []
    img_info = dict()
    img_info["imageHeight"] = data.get("imageHeight")
    img_info["imageWidth"] = data.get("imageWidth")
    for old_info in data["shapes"]:
        bbox = old_info["points"][0] + old_info["points"][1]
        label = old_info["label"]
        track_id = old_info["group_id"]
        info = dict(
            bbox=bbox,
            label=label,
            track_id=track_id,
        )
        if "attribute" in old_info:
            info["attribute"] = old_info["attribute"]
        if "score" in old_info:
            info["score"] = old_info["score"]
        info_list.append(info)
    return info_list, img_info


def load_anno(path):
    if not os.path.exists(path):
        raise ValueError("not exist", path)
    data = load_json(path)
    info_list = []
    for old_info in data["DataList"]:
        bbox = old_info["bbox"]
        label = old_info["label"]
        info = dict(
            bbox=bbox,
            label=label,
        )
        if "track_id" in old_info:
            info["track_id"] = old_info["track_id"]
        if "event_type" in old_info:
            info["event_type"] = old_info["event_type"]
        info_list.append(info)
    return info_list


def load_L1(path, img_w=1920, img_h=1080):
    DIC = {
        0: "car",
        1: "bus",
        2: "truck",
        3: "人车or车牌",
        4: "pedestrian",
        5: "bike",
        6: "motor",
        7: "tricycle",
        8: "rider",
        9: "head",
        10: "rider_with_motor",
    }

    if not os.path.exists(path):
        raise ValueError("not exist", path)
    lines = load_txt(path)
    print(path)
    info_list = []
    for line in lines:
        line = line.split(";")[0]
        line = line.split(" ")
        while "" in line:
            line.remove("")
        label, x_c, y_c, w, h, trk_id, *args = [float(x) for x in line]
        x_min = (x_c - w / 2) * img_w
        y_min = (y_c - h / 2) * img_h
        x_max = (x_c + w / 2) * img_w
        y_max = (y_c + h / 2) * img_h
        bbox = [int(x) for x in [x_min, y_min, x_max, y_max]]
        label = int(label)
        track_id = int(trk_id)
        info = dict(
            bbox=bbox,
            label=DIC[label],
            track_id=track_id,
        )
        if args:
            score = args[0]
            info["score"] = score
        info_list.append(info)
    return info_list


def load_yolo_txt(path, img_w, img_h):
    if not os.path.exists(path):
        warn("not exist {}".format(path))
        return []
    lines = load_txt(path)
    info_list = []
    for line in lines:
        line = line.split(" ")
        while "" in line:
            line.remove("")
        label, x_c, y_c, w, h, *args = [float(x) for x in line]
        x_min = (x_c - w / 2) * img_w
        y_min = (y_c - h / 2) * img_h
        x_max = (x_c + w / 2) * img_w
        y_max = (y_c + h / 2) * img_h
        bbox = [int(x) for x in [x_min, y_min, x_max, y_max]]
        label = int(label)
        info = dict(
            bbox=bbox,
            label=label,
        )
        info_list.append(info)
    return info_list


def load_cvires_txt(path, score_thre=0.4):
    if not os.path.exists(path):
        warn("not exist {}".format(path))
        return []
    lines = load_txt(path)
    info_list = []
    for line in lines:
        line = line.split(" ")
        while "" in line:
            line.remove("")
        label, x_min, y_min, x_max, y_max, score = [float(x) for x in line]
        if score < score_thre:
            continue
        bbox = [int(x) for x in [x_min, y_min, x_max, y_max]]
        label = int(label)
        info = dict(
            bbox=bbox,
            label=label,
        )
        info_list.append(info)
    return info_list