import json
import os


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except:
            print(file_path)
            raise e
    return data


def load_labelme(path):
    if not os.path.exists(path):
        raise ValueError("not exist", path)
    data = load_json(path)
    info_list = []
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
    return info_list


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