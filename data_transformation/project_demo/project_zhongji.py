from data_transformation import DataTransforms

'''
content:
    image:
    info:
        imageName:
        imageWidth:
        imageHeight:
        imageDepth: 
        shapes:[
            {
                shape_type: "polygon",  # [polygon | bndbox]
                label: str,
                points': [(693, 224), (739, 253), ...],  # point1, point2, ...
            },
            {
                'shape_type': "bndbox",
                'label': 'hand', 
                'points': [692, 254, 709, 275],  # x_min, y_min, x_max, y_max.
            },
            ...
        ]
'''

import numpy as np


class EmptyMask:
    def process(self, content):
        h, w = content["image"].shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        content["image"] = mask
        return content


if __name__ == '__main__':
    label_val_dict = {"gansha": 1,
                      "kailie": 2,
                      "huangban": 3,
                      "xuhua": 4,
                      "wuwu": 5,
                      "bujun": 6,
                      "wanzhe": 7,
                      }
    transforms = DataTransforms(reader_method={"image": "/run/user/1000/gvfs/smb-share:server=10.18.103.158,share=share/项目/中集CFRT/数据/现场采集数据/2021-07-15/data/0714/开裂错检",
                                               },
                                writer_method={"image": "/run/user/1000/gvfs/smb-share:server=10.18.103.158,share=share/项目/中集CFRT/数据/现场采集数据/2021-07-15/data/0714/开裂错检_mask",
                                               },
                                processor="anno2mask",
                                )
    transforms.apply()
