!nvidia-smi
!pip install ultralytics
!pip install roboflow

--------------------------

from ultralytics import YOLO

model = YOLO("yolov10n.yaml")

model_train = model.train(data="/content/Nasa-SpaceApps-2/data.yaml", epochs=50, imgsz=640, batch=16)

--------------------------



model = YOLO("/content/Nasa-SpaceApps-2/best.pt")  # load a custom model

metrics = model.val()
metrics.box.map
metrics.box.map50 
metrics.box.map75
metrics.box.maps

--------------------------



model = YOLO("/content/Nasa-SpaceApps-2/best.pt")

results = model(["image1.jpg", "image2.jpg"]) 

for result in results:
    boxes = result.boxes 
    masks = result.masks  
    keypoints = result.keypoints  
    probs = result.probs 
    obb = result.obb  
    result.show() 
    result.save(filename="result.jpg")
