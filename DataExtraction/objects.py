
import sys
sys.path.append('yolo')
sys.path.append('DataExtraction')

from objectDetection import detect_objects_in_all_frames
from save import saveData

import torch
def loadYOLOv5():
    # Load the model
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True,)

    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    return yolo_model,classes

def detectObjects(frames, objects=None,encoded_objects=None,video=None,tokenizer=None):
    if objects is None:
        print('Detecting objects in frames...')
        yolo_model, classes = loadYOLOv5()
        
        objects = detect_objects_in_all_frames(frames, yolo_model, classes)
        saveData('objects',objects,video)

    # Here, taking the first detected object
    objects = [frame_objects if frame_objects else ['None'] for frame_objects in objects]

    count=0
    for obj in objects:
        for o in obj:
            count+=1
            
    # print(f'Total Encoded Obj: {count}')
    # print("Objects:",len(objects))

    # One-hot encoding of objects
    if encoded_objects is None:
        encoded_objects = []
        for frame_objects in objects:
            # Encode each object in the frame
            encoded_frame_objects = [tokenizer(ob).vector for ob in frame_objects]
            # Add the list of encoded objects for this frame to the main list
            encoded_objects.append(encoded_frame_objects)
        
    
    return encoded_objects, objects