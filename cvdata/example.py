from cvdata import DataTransforms
from cvdata.processor import Anno2MaskProcessor

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


class GrayProcessor:
    def process(self, content):
        """do some thing"""
        return content


class ModelResult:
    def __init__(self):
        self.model = ...

    def process(self, content):
        res = self.model.infer(content["image"])
        content["info"] = ...(res)
        return content


if __name__ == '__main__':
    label_val_dict = {"qikong": 1, "shusong": 2}
    transforms = DataTransforms(
        reader_method={
            "image": "/run/user/1000/gvfs/smb-share:server=10.18.103.158,share=share/项目/无损探伤/数据/现场采集/2021-07-19/气孔tiff",
            # "xml": "/run/user/1000/gvfs/smb-share:server=10.18.103.158,share=share/项目/无损探伤/数据/标注/2021-07-12"
            },
        writer_method={
            "image": "/run/user/1000/gvfs/smb-share:server=10.18.103.158,share=share/项目/无损探伤/数据/现场采集/2021-07-19/气孔jpg",
            # "xml": "../test_imgs/outputs/xml/"
            },
        processor="img2jpg",  # Anno2MaskProcessor(label_val_dict=label_val_dict),
    )
    transforms.apply()
