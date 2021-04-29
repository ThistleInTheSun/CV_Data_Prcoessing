#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/27
# @Author  : xq

if __name__ == '__main__':
    from manuvision_img_utils.core.io.reader import Reader
    from manuvision_img_utils.core.io.writer import Writer
    from tqdm import tqdm
    import os

    imgR = Reader("/home/xq/文档/projects/安洁电子/data/3M胶数据-ROI_rotated_ok_dirty-line/mask", "GRAY")
    resW = Writer(r'/home/xq/文档/projects/安洁电子/data/3M胶数据-ROI_rotated_ok_dirty-line/mask')

    for fn in tqdm(imgR.filenames):
        img = imgR.get_image(fn)
        fn = os.path.splitext(fn)[0]
        fn_now_img = "{}.png".format(fn)
        resW.save(img, fn_now_img)



