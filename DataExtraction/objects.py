
import sys
sys.path.append('yolo')
sys.path.append('DataExtraction')
sys.path.append('model')
from objectDetection import detect_objects_in_all_frames
from save import saveData
from object_embedding_model import process_detected_objects


import torch
def loadYOLOv5():
    # Load the model
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True,)

    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    return yolo_model,classes

def detectObjects(frames, objects=None,encoded_objects=None,video=None,tokenizer=None):
    yolo_model, classes = loadYOLOv5()
    if objects is None:
        print('Detecting objects in frames...')
        
        objects = detect_objects_in_all_frames(frames, yolo_model, classes)
        saveData('objects',objects,video)

    # Here, taking the first detected object
    objects = [frame_objects if frame_objects else [] for frame_objects in objects]
    
    if encoded_objects is None:
        print('Processing detected objects...')
        encoded_objects = process_detected_objects(objects,classes)
    
    print('Detected objects processed.')
    
    return encoded_objects, objects