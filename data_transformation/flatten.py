import os
import shutil


def flatten(source, target):
    imgs = os.listdir(source)
    for sub_file in imgs:
        if not os.path.isdir(os.path.join(source, sub_file)):
            print("source:", os.path.join(source, sub_file))
            print("target:", os.path.join(target, sub_file))
            shutil.copy(os.path.join(source, sub_file), os.path.join(target, sub_file))
        else:
            sub_imgs = os.listdir(os.path.join(source, sub_file))
            for sub_sub_file in sub_imgs:
                shutil.copy(os.path.join(source, sub_file, sub_sub_file), os.path.join(target, sub_file + "_" + sub_sub_file))


if __name__ == '__main__':
    source = "/home/xq/文档/projects/日联/data/2021-08-06"
    target = "/home/xq/文档/projects/日联/data/2021-08-06_flatten"
    flatten(source, target)


