#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/25
# @Author  : xq
import os
import random
import shutil


IMG_SUFFIX_LIST = [".jpg", ".jpeg", ".JPG", ".JPEG"]


def random_selected(path, write_path):
    os.makedirs(write_path, exist_ok=True)
    img_list = [i for i in os.listdir(path) if os.path.splitext(i)[-1] in IMG_SUFFIX_LIST]
    print("all:", len(img_list))
    # rate = 0.2
    random.seed(42)
    selected = random.sample(img_list, 500)
    print("selected:", len(selected), "others:", len(img_list) - len(selected))
    for img_name in selected:
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        shutil.move(os.path.join(path, img_name), os.path.join(write_path, img_name))
        shutil.move(os.path.join(path, xml_name), os.path.join(write_path, xml_name))


if __name__ == "__main__":
    # read_path = '/opt/data/public02/manutyh/xiangqing/jingxia/data/visible_light/model_xml_res_2021-04-14_v'
    # write_path = '/opt/data/public02/manutyh/xiangqing/jingxia/data/visible_light/model_xml_res_2021-04-14_v_selected'

    read_path = '/opt/data/public02/manutyh/xiangqing/jingxia/data/termal_image/model_res/model_xml_res_in_disc_3'
    write_path = '/opt/data/public02/manutyh/xiangqing/jingxia/data/termal_image/model_res/model_xml_res_in_disc_3_selected500'

    random_selected(read_path, write_path)



