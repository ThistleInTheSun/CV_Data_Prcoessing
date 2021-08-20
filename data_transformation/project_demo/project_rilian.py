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


class RenameProcess:
    def process(self, content):
        h, w = content["image"].shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        content["image"] = mask
        return content


if __name__ == '__main__':
    transforms = DataTransforms(reader_method={"image": "/home/xq/文档/projects/日联/data/2021-08-06_flatten",
                                               },
                                writer_method={"image": "/home/xq/文档/projects/日联/data/2021-08-06_flatten_jpg",
                                               },
                                processor="img2jpg",
                                )
    transforms.apply()
