from data_transformation import DataTransforms
from data_transformation.processor import Anno2MaskProcessor

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


if __name__ == '__main__':
    label_val_dict = {"gansha": 1,
                      "kailie": 2,
                      "huangban": 3,
                      "xuhua": 4,
                      "wuwu": 5,
                      "bujun": 6,
                      "wanzhe": 7,
                      }
    transforms = DataTransforms(reader_method={"image": "/run/user/1000/gvfs/smb-share:server=10.18.103.158,share=share/项目/中集CFRT/标注/标注结果/...",
                                               "xml": "/run/user/1000/gvfs/smb-share:server=10.18.103.158,share=share/项目/中集CFRT/标注/标注结果/..."},
                                writer_method={"image": "/run/user/1000/gvfs/smb-share:server=10.18.103.158,share=share/项目/中集CFRT/标注/标注结果/.../mask",
                                               },
                                processor=Anno2MaskProcessor(label_val_dict=label_val_dict),
                                )
    transforms.apply()
