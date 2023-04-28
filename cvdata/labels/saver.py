import os
import json
import os
import cv2

from cvdata.labels.visualize import plot_tracking


class JsonSaver():
    def __init__(self, save_dir) -> None:
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, res, name, img=None, w=None, h=None, *args, **kwargs):
        lb_data = {"shapes": []}
        for info in res:
            box = info["bbox"]
            save_info = {
                "label": info["label"],
                "points": [[int(box[0]), int(box[1])], [int(box[2]), int(box[3])]],
                "shape_type": "rectangle",
                "group_id": info["track_id"] if "track_id" in info else -1,
            }
            if "attribute" in info:
                save_info["attribute"] = info["attribute"]
            if "score" in info:
                save_info["score"] = info["score"]
            lb_data["shapes"].append(save_info)
        if img is not None:
            h, w = img.shape[:2]
        if w is not None and h is not None:
            lb_data["imageHeight"] = h
            lb_data["imageWidth"] = w
        if not name.endswith(".json"):
            name += ".json"
        save_path = os.path.join(self.save_dir, name)
        if res:
            with open(save_path, "w") as f:
                json.dump(lb_data, f, indent=2, sort_keys=True)


class ImgSaver():
    def __init__(self, save_dir, is_draw=True) -> None:
        self.save_dir = save_dir
        self.is_draw = is_draw
        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, res, name, img, *args, **kwargs):
        if self.is_draw:
            img = plot_tracking(img, res)
        if not name.endswith(".jpg"):
            name += ".jpg"
        save_path = os.path.join(self.save_dir, name)
        cv2.imwrite(save_path, img)


class VideoSaver():
    def __init__(self, save_dir, fps=25, width=1920, height=1080, is_draw=True) -> None:
        self.save_dir = save_dir if save_dir.endswith(".mp4") else save_dir + ".mp4"
        self.width = width
        self.height = height
        self.is_draw = is_draw
        self.vid_writer = cv2.VideoWriter(
            self.save_dir, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        path_dir = os.path.split(save_dir)[0]
        os.makedirs(path_dir, exist_ok=True)
        print(self.save_dir)

    def __call__(self, res, img, *args, **kwargs):
        if img is None:
            raise ValueError("img is none")
        if self.is_draw:
            img = plot_tracking(img, res)
        img = cv2.resize(img, (self.width, self.height))
        self.vid_writer.write(img)