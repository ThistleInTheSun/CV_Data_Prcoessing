import os
import shutil
import sys
from warnings import warn

import cv2
from tqdm import tqdm

sys.path.append(os.getcwd())
from cvdata.labels.saver import save_txt

'''00227490421455-90_88-341&443_436&482-437&479_338&486_335&452_434&445-0_0_9_33_33_29_28-128-12.txt

0 0.6125  0.3065  0.5750  0.2905 1;0.8319 0.1802 0.8097 0.1672 0.7861 0.1603 0.6861 0.1621 0.6458 0.1681 0.5819 0.1707 0.5472 0.1784 0.5222 0.1810 0.4750 0.2017 0.4528 0.2155 0.4458 0.2233 0.4319 0.2319 0.4250 0.2422 0.4139 0.2491 0.4042 0.2603 0.3931 0.2655 0.3847 0.2793 0.3736 0.2871 0.3653 0.3009 0.3528 0.3095 0.3458 0.3198 0.3292 0.3612 0.3292 0.3922 0.3333 0.4078 0.3528 0.4328 0.3778 0.4414 0.4056 0.4448 0.4889 0.4466 0.5889 0.4448 0.6528 0.4405 0.7319 0.4388 0.8028 0.4336 0.8278 0.4284 0.8347 0.4233 0.8417 0.3914 0.8542 0.3759 0.8597 0.3629 0.8903 0.3405 0.8958 0.3241 0.8958 0.2664 0.8903 0.2474 0.8806 0.2371 0.8750 0.2224 0.8625 0.2129 0.8528 0.2000
0 0.9556  0.1901  0.0833  0.0836 2;0.9986 0.1552 0.9778 0.1552 0.9611 0.1595 0.9361 0.1767 0.9194 0.1983 0.9194 0.2216 0.9236 0.2310 0.9583 0.2310
3 0.5396  0.3987  0.1319  0.0336 3
6 0.2838  0.2845  0.2291  0.1110 4;0.3903 0.2647 0.3556 0.2509 0.3569 0.2414 0.3542 0.2371 0.3375 0.2310 0.3181 0.2319 0.2931 0.2388 0.2847 0.2466 0.2847 0.2509 0.2931 0.2552 0.2875 0.2621 0.2806 0.2655 0.2625 0.2664 0.2194 0.2612 0.1924 0.2621 0.1903 0.2629 0.1944 0.2741 0.1889 0.2862 0.1750 0.2974 0.1806 0.3078 0.1819 0.3259 0.1972 0.3353 0.2208 0.3379 0.2333 0.3353 0.2500 0.3259 0.2639 0.3267 0.3069 0.3198 0.3361 0.3198 0.3514 0.3112 0.3667 0.2983
6 0.0566  0.3328  0.1131  0.1638 5

'''
def L12yolo(img_dir, txt_dir, save_dir):
    shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    txt_list = sorted(os.listdir(txt_dir))
    for txt_name in tqdm(txt_list):
        L1_path = os.path.join(txt_dir, txt_name)
        lines = []
        with open(L1_path) as f:
            for line in f:
                line = line.split(";")[0]
                line = line.split(" ")
                while "" in line:
                    line.remove("")
                label, x, y, w, h, *args = line
                if y == '':
                    print("sorce:", line)
                    raise
                if int(label) not in [4, 8, 9]:
                    continue
                if int(label) in [4, 8]:  # person
                    label = 1
                if int(label) in [9]:  # head
                    label = 0
                line = " ".join([str(label), x, y, w, h])
                lines.append(line)
        save_path = os.path.join(save_dir, txt_name)
        save_txt(save_path, lines)


if __name__ == "__main__":
    for dir_name in ["ccpd", "soda10m_train", "soda10m_val", "videoimg"]:
        img_dir = "/dataset/head_person_train/{}/images".format(dir_name)
        txt_dir = "/dataset/head_person_train/{}/source_labels".format(dir_name)
        save_dir = "/dataset/head_person_train/{}/labels".format(dir_name)
        L12yolo(img_dir, txt_dir, save_dir)

