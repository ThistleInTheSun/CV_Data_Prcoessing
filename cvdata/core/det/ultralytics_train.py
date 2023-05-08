from ultralytics import YOLO

# Load a model
model = YOLO("./config/yolov8l.yaml")  # build a new model from scratch
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="./config/cell_phone.yaml", epochs=100)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# model.predict("/home/sdb1/eco_algo_images/data_scene005_00016_female_Asian_20_01_091") 
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format


model = YOLO("./config/yolov8l.yaml")  # build a new model from scratch
model = YOLO("/home/qing.xiang/algorithm/yolov8_train/runs/detect/train20/weights/best.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx")  # export the model to ONNX format