import unittest
import cv2


class MyTestCase(unittest.TestCase):
    def setUp(self):
        img = cv2.imread("../test_imgs/inputs/ch02_20210315161553_00041250.jpg")
        self.content = {"image": img,
                        'info': {
                            'imageName': 'ch02_20210315161553_00041300.jpg',
                            'imageWidth': '1920',
                            'imageHeight': '1080',
                            'imageDepth': '3',
                            'shapes': [{'shape_type': 'bndbox', 'label': 'hand', 'points': [693, 224, 739, 253]},
                                       {'shape_type': 'bndbox', 'label': 'hand', 'points': [692, 254, 709, 275]},
                                       {'shape_type': 'polygon', 'label': 'hand', 'points': [(1048, 564), (1087, 600)]},
                                       {'shape_type': 'polygon', 'label': 'hand', 'points': [(1145, 581), (1202, 616)]},
                                       ]
                        }}

    def test_ImageWriter(self):
        from data_transformation.writer import ImageWriter
        writer = ImageWriter("../test_imgs/outputs/")
        writer.update_path("test_sub")
        writer.write(self.content)

    def test_XmlWriter(self):
        from data_transformation.writer import XmlWriter
        writer = XmlWriter("../test_imgs/outputs/")
        writer.write(self.content)

    def test_TxtWriter(self):
        from data_transformation.writer import TxtWriter
        writer = TxtWriter("../test_imgs/outputs/")
        writer.write(self.content)


if __name__ == '__main__':
    unittest.main()
