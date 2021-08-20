import unittest
from unittest import TestCase
import os


class TestReader(unittest.TestCase):
    # def test_tmp(self):
    #     d = os.path.join("ab", "c")
    #     print(d)

    def test_content_reader(self):
        from data_transformation.core.reader import ImageReader, ConcatReader
        reader1 = ImageReader("../test_imgs/inputs/")
        c_reader = ConcatReader([reader1], is_recursive=False)
        for i, content in enumerate(c_reader):
            pass
        self.assertEqual(i, 4)

    def test_content_is_recursive(self):
        from data_transformation.core.reader import ImageReader, ConcatReader
        reader1 = ImageReader("../test_imgs/inputs/")
        c_reader = ConcatReader([reader1], is_recursive=True)
        for i, content in enumerate(c_reader):
            if "ptype" not in content["info"]:
                print("no ptype in", content["info"]["imageName"])
                continue
            print("ptype:", content["info"]["ptype"])
        self.assertEqual(i, 34)

    def test_ImgReader(self):
        from data_transformation.core.reader import ImageReader
        reader = ImageReader("../test_imgs/inputs/")
        self.assertIn("image", reader[0])
        self.assertIn("info", reader[0])
        self.assertIn("imageName", reader[0]["info"])

    def test_VideoReader(self):
        from data_transformation.core.reader import VideoReader
        reader = VideoReader("../test_imgs/inputs/video.mp4")
        self.assertEqual(reader.video_name, "video")
        self.assertEqual(len(reader), 284)
        for content in reader:
            self.assertIn("image", content)
            self.assertIn("info", content)
            self.assertIn("imageName", content["info"])
            break

    def test_JsonReader(self):
        from data_transformation.core.reader import JsonReader
        reader = JsonReader("../test_imgs/inputs/img_and_json")
        self.assertIn("info", reader[0])
        self.assertIn("shapes", reader[0]["info"])

    def test_XmlReader(self):
        from data_transformation.core.reader import XmlReader
        reader = XmlReader("../test_imgs/inputs/img_and_xml")
        self.assertIn("info", reader[0])
        self.assertIn("shapes", reader[0]["info"])


if __name__ == '__main__':
    unittest.main()
