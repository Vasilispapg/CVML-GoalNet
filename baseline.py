import csv
import numpy as np
from sklearn.metrics import mean_squared_error


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


annotation_fp = 'ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv'

video_fp = './ydata-tvsum50-v1_1/video/VuWGsYPqAX8.mp4'
ground_truth = get_annotations(annotation_fp = annotation_fp, video_id = video_fp.split('/')[-1].split('.')[0])
mse_constant = mean_squared_error(ground_truth, [3]*len(ground_truth))
print('Baseline (sample VuWGsYPqAX8) - Random regressor: %.4f'%(mse_constant))