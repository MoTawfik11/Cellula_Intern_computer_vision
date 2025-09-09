from ultralytics import YOLO

model = YOLO(r"E:\\Computer Vision\\Cellula Tech\\Task 2\\runs\\classify\\train\\weights\\best.pt")
model.export(format="onnx", opset=12, dynamic=True)
