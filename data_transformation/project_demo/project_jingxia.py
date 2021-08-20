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


class ModelResult:
    def __init__(self):
        self.model = ...

    def process(self, content):
        res = self.model.infer(content["image"])
        content["info"] = ...(res)
        return content


if __name__ == '__main__':
    transforms = DataTransforms(
        reader_method={
            "video": "/opt/data/public02/manutyh/xiangqing/jingxia/data/termal_image/video/2021-07-21/",
            # "xml": "/run/user/1000/gvfs/smb-share:server=10.18.103.158,share=share/项目/无损探伤/数据/标注/2021-07-12"
            },
        writer_method={
            "video": "/opt/data/public02/manutyh/xiangqing/jingxia/data/termal_image/video/2021-07-21_res/",
            # "xml": "../test_imgs/outputs/xml/"
            },
        processor=[ModelResult(), "draw"],
    )
    transforms.apply()
