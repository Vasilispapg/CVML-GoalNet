import csv

def getAnnotations(annotation_path, videoID):
    annotations = []
    with open(annotation_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        
        for row in reader:
            if row[0] == videoID:
                annotations.append(row[2].strip().split(','))
                
    "annotations = 20 annotations for each frame"
    # shape of annotations: (20, frames)
    # get mean for each frame
    # and get each 1 each 15 frames
    annotations=np.array(annotations, dtype = np.float32).T
    mean_annotations=[]
    count=0
    for ann in annotations:
        if(count%15==0):
            mean_annotations.append(np.mean(ann))
        count+=1
    mean_annotations = np.array(mean_annotations)
    # TODO : may use multi label later for each frame
    
    return mean_annotations


import numpy as np

def extract_frames(video_path, frame_rate=15):
    video = cv2.VideoCapture(video_path)
    count = 0
    success = True
    frames = []
    
    while success:
        success, image = video.read()
        if count % frame_rate == 0 and success:
            # resize image to 299x299
            image = ((image - image.min()) / (image.max() - image.min())).astype(np.float32)
            image = cv2.resize(image, (299, 299))
            frames.append(image)
            
        count += 1

    video.release()
    return frames