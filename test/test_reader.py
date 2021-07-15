import unittest


class MyTestCase(unittest.TestCase):
    def test_is_recursive(self):
        from data_transformation.reader import ImageReader
        reader1 = ImageReader("../test_imgs/inputs/")
        self.assertEqual(len(reader1), 5)
        reader2 = ImageReader("../test_imgs/inputs/", is_recursive=True)
        self.assertEqual(len(reader2), 20)

    def test_ImgReader(self):
        from data_transformation.reader import ImageReader
        reader = ImageReader("../test_imgs/inputs/")
        self.assertIn("image", reader[0])
        self.assertIn("info", reader[0])
        self.assertIn("imageName", reader[0]["info"])

    def test_VideoReader(self):
        from data_transformation.reader import VideoReader
        reader = VideoReader("../test_imgs/inputs/video.mp4")
        self.assertEqual(reader.video_name, "video")
        self.assertEqual(len(reader), 284)
        for content in reader:
            self.assertIn("image", content)
            self.assertIn("info", content)
            self.assertIn("imageName", content["info"])
            break

    def test_JsonReader(self):
        from data_transformation.reader import JsonReader
        reader = JsonReader("../test_imgs/inputs/img_and_json")
        self.assertIn("info", reader[0])
        self.assertIn("shapes", reader[0]["info"])

    def test_XmlReader(self):
        from data_transformation.reader import XmlReader
        reader = XmlReader("../test_imgs/inputs/img_and_xml")
        self.assertIn("info", reader[0])
        self.assertIn("shapes", reader[0]["info"])


if __name__ == '__main__':
    unittest.main()
