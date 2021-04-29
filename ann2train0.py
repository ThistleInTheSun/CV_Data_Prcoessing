import os
import cv2
import json
import tqdm
import random
from xml.dom.minidom import parse
from shutil import copyfile
from collections import defaultdict


IMG_SUFFIX_LIST = [".jpg", ".jpeg", ".JPG", ".JPEG"]


def anno2minmax(shape, w, h):
    x1 = max(min(shape['points'][0][0], shape['points'][1][0]), 0)
    x2 = min(max(shape['points'][0][0], shape['points'][1][0]), w)
    y1 = max(min(shape['points'][0][1], shape['points'][1][1]), 0)
    y2 = min(max(shape['points'][0][1], shape['points'][1][1]), h)
    return x1, x2, y1, y2


def load_json(path):
    if os.path.splitext(path)[-1] != ".json":
        print("Cannot find {}".format(path))
        return
    json_file = open(path)
    json_cont = json.load(json_file)
    res = []
    for obj in json_cont["shapes"]:
        cur_obj = {}
        cur_obj["label"] = obj["label"]
        cur_obj["points"] = obj["points"][0] + obj["points"][1]
        res.append(cur_obj)
    json_file.close()
    return res


def load_xml(path):
    if os.path.splitext(path)[-1] != ".xml":
        print("Cannot find {}".format(path))
        return
    res = []
    dom_tree = parse(path)
    root_node = dom_tree.documentElement
    objects = root_node.getElementsByTagName("object")
    for obj in objects:
        cur_obj = dict()
        cur_obj["label"] = obj.getElementsByTagName("name")[0].childNodes[0].data
        cur_obj["points"] = [int(obj.getElementsByTagName("bndbox")[0].getElementsByTagName(point)[0].childNodes[0].data) \
                             for point in ["xmin", "ymin", "xmax", "ymax"]]
        res.append(cur_obj)
    return res


def convert_xxyy2xywh(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def anno2minmax(obj, w, h):
    p0, p1, p2, p3 = obj["points"]  # x1, y1, x2, y2
    x_min = max(min(p0, p2), 0)
    x_max = min(max(p0, p2), w)
    y_min = max(min(p1, p3), 0)
    y_max = min(max(p1, p3), h)

    b = (float(x_min), float(x_max), float(y_min), float(y_max))
    points = convert_xxyy2xywh((w, h), b)
    return points


ANNOTATION_SUFFIX = {".json": load_json,
                     ".xml": load_xml}


def remove_illegal_name(file_id):
    return os.path.splitext(file_id)[0].replace(".", "").replace(" ", "")


class Json2Yolov4(object):
    def __init__(self, ain_path, target_path, classes_dict=None, ratio=0.9, mod="json2mask"):
        self.ain_path = ain_path
        self.target_path = target_path
        self.is_limit_cls = bool(classes_dict)
        self.classes_dict = {} if classes_dict is None else classes_dict
        self.ratio = ratio

        self.name = ain_path.split("/")[-1]
        self.classNums = defaultdict(int)
        self.train_list = []
        self.valid_list = []
        self.none_json = 0
        self.num = 0

        if os.path.exists(self.target_path):
            print("warning: {} has been exist.".format(self.target_path))
        else:
            os.makedirs(self.target_path)
            os.makedirs(os.path.join(self.target_path, "images"))
            os.makedirs(os.path.join(self.target_path, "labels"))

    def get_label(self, label):
        if self.is_limit_cls:
            for pre_label in self.classes_dict.keys():
                if pre_label in label:
                    label = pre_label
                    break
            else:
                return None
        elif label not in self.classes_dict:
            self.classes_dict[label] = len(self.classes_dict)
        self.classNums[label] += 1
        cls_id = self.classes_dict[label]
        return cls_id

    def write_anno_txt(self, out_txt, obj, w, h):
        cls_id = self.get_label(obj["label"])
        if cls_id is None:
            return
        points = anno2minmax(obj, w, h)
        out_txt.write(str(cls_id) + " " + " ".join([str(p) for p in points]) + '\n')

    def convert_json_annotation(self, ain_path, file, src_name, tar_name):
        img_path = os.path.join(ain_path, file)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        if h > w:
            h, w = w, h

        for suffix in ANNOTATION_SUFFIX.keys():
            if os.path.exists(os.path.join(ain_path, src_name + suffix)):
                load_anno = ANNOTATION_SUFFIX[suffix]
                anno_path = os.path.join(ain_path, src_name + suffix)
                break
        anno_content = load_anno(anno_path)
        out_txt = open(os.path.join(self.target_path, "labels", tar_name + '.txt'), 'w')
        for obj in anno_content:
            self.write_anno_txt(out_txt, obj, w, h)  # obj["points"]: x_min, y_min, x_max, y_max
        out_txt.close()

    def do_one_img(self, ain_path, file, src_name, tar_name):
        src_img = os.path.join(ain_path, file)
        tar_img = os.path.join(self.target_path, "images", tar_name + ".jpg")
        if os.path.exists(os.path.join(ain_path, src_name + ".json"))\
                or os.path.exists(os.path.join(ain_path, src_name + ".xml")):
            self.num += 1
            copyfile(src_img, tar_img)
            self.convert_json_annotation(ain_path, file, src_name, tar_name)
        else:
            self.none_json += 1
            return False
        return True

    def apply(self, ain_path):
        file_list = os.listdir(ain_path)
        for file in file_list:
            if os.path.isdir(os.path.join(ain_path, file)):
                sub_ain_path = os.path.join(ain_path, file)
                self.apply(sub_ain_path)
                print(file, "done")
            elif os.path.splitext(file)[-1] in IMG_SUFFIX_LIST:
                src_name = os.path.splitext(file)[0]
                tar_name = remove_illegal_name(file)
                has_ain = self.do_one_img(ain_path, file, src_name, tar_name)
                if not has_ain:
                    continue

                idx = self.target_path.find("data")
                img_path = os.path.join(self.target_path[idx:], "images", tar_name + ".jpg")
                if random.randint(0, 100) < self.ratio * 100:
                    self.train_list.append(img_path)
                else:
                    self.valid_list.append(img_path)

    def write_data(self):
        idx = self.target_path.find("data")
        save_path = os.path.join(self.target_path, self.name + ".data")
        f = open(save_path, "w")
        f.write("classes={}\n".format(len(self.classNums)))
        f.write("train={}\n".format(os.path.join(self.target_path[idx:], "train.txt").replace("\\", "/")))
        f.write("valid={}\n".format(os.path.join(self.target_path[idx:], "valid.txt").replace("\\", "/")))
        f.write("names={}\n".format(os.path.join(self.target_path[idx:], self.name + ".names").replace("\\", "/")))
        f.write("backup=backup/\n")
        f.close()

    def write_names(self):
        save_path = os.path.join(self.target_path, self.name + ".names")
        f = open(save_path, "w")
        for i in self.classNums.keys():
            f.write("{}\n".format(i))
        f.close()

    def write_classNums(self):
        save_path = os.path.join(self.target_path, "classNums.txt")
        f = open(save_path, "w")
        for i in self.classes_dict.keys():
            f.write("%s : %d\n" % (i, self.classNums[i]))
        f.close()

    def write_train_valid_txt(self):
        train_txt = open(os.path.join(self.target_path, 'train.txt'), 'w')
        for img_path in self.train_list:
            train_txt.write("%s\n" % img_path)
        train_txt.close()

        valid_txt = open(os.path.join(self.target_path, 'valid.txt'), 'w')
        for img_path in self.valid_list:
            valid_txt.write("%s\n" % img_path)
        valid_txt.close()


def main(ain_path, target_path, classes_dict, ratio):
    json2yolov4 = Json2Yolov4(ain_path, target_path, classes_dict, ratio)
    json2yolov4.apply(ain_path)
    json2yolov4.write_data()
    json2yolov4.write_names()
    json2yolov4.write_classNums()
    json2yolov4.write_train_valid_txt()
    print("no labeled image: {}".format(json2yolov4.none_json))
    print("train", len(json2yolov4.train_list))
    print("valid", len(json2yolov4.valid_list))
    print("class", json2yolov4.classNums)


if __name__ == '__main__':
    # ain_path = '/home/h3c/xq/pytorch_yolov4/data/annotation_cell_person'
    # target_path = '/home/h3c/xq/pytorch_yolov4/data/annotation_cell_person_datasets_24+26+30+09'
    ain_path = "/home/xq/文档/projects/加油站/data/便利店数据-已完成"
    target_path = "/home/xq/文档/projects/加油站/data/annotation_gas_station_store_sh_datasets"
    # classes_dict = None
    mod = "json2mask"
    classes_dict = {"person": 0, "hand": 1, "phone": 2}
    ratio = 1
    main(ain_path=ain_path, target_path=target_path, classes_dict=classes_dict, ratio=ratio, mod=mod)


