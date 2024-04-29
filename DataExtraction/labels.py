import csv
import numpy as np


def get_annotations(annotation_fp, video_id):
    annotations = []
    with open(annotation_fp, 'r') as file:
        reader = csv.reader(file, delimiter='\t')

        for row in reader:
            if row[0] == video_id:
                annotations.append(row[2].strip().split(','))

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

    return mean_annotations