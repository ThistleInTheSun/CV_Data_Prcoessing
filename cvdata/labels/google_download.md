# 方法一：ubuntu系统安装google-chrome插件
# 方法二：图片助手（ImageAssistant）【推荐！chrome插件安装更方便，下载的图片也更多】
查看是否有chrome：
有：查看版本：google-chrome --version
无：安装然后查看版本：sudo dpkg -i google-chrome*.deb

# 
根据上述版本，下载对应版本的chromedriver
加压后把chromedriver放到*/bin/.下面


# down google img: (248 server)
cd /home/sdb1/xq/taiwan/data/object_detection/coco_wider_pedestrian/google_images

cd /dataset/object_detection/coco_wider_pedestrian
python Image-Downloader/image_downloader.py \
"半裸男" \
--max-number 10000 \
--output data_nude_1 \
--driver chrome_headless \
--timeout 500


# 报错
AttributeError: type object 'DesiredCapabilities' has no attribute 'PHANTOMJS'
版本太高，降版
pip uninstall selenium
pip install selenium==2.48.0