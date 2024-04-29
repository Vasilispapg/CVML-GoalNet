
import sys
sys.path.append('DataExtraction')
sys.path.append('Evaluation')
sys.path.append('videoSum')
sys.path.append('knapsack')
import os
import numpy as np

from frames import extract_frames
from mapping import map_scores_to_original_frames
from clipImportance import getClipImportances
from clipImportance import getSelectedIndicesFromClips
from knapsack import knapsack_for_video_summary
from fscoreEval import evaluation_method
from deletePkl import deletePKLfiles
from videoCreator import create_video_from_frames
from data import DataAudioExtraction
from model import callNN
from labels import getAnnotations
import random

annotation_path='datasets/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv' # Path for importantce (ground truth (partial))
info_path='datasets/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv' # Path for the videos (e.g. links)
video_path='datasets/ydata-tvsum50-v1_1/video/' # Input videos
summary_video_path='datasets/summary_videos/' # Output pou vazoume ta video summaries
ground_truth_path='datasets/ydata-tvsum50-v1_1/ground_truth/ydata-tvsum50.mat' # Katigoria annotation users etc (ground truth partial)
video_list = [video for video in os.listdir(video_path) if video.endswith('.mp4')]  # List comprehension

# extract the frames from the test video
test_video= video_list.pop(random.randint(0,len(video_list)-1))  # get as test video a random video
test_dataset= [extract_frames(video_path+test_video), 
               DataAudioExtraction(video_path+test_video,anno_file=annotation_path,info_file=info_path),
               getAnnotations(annotation_path, test_video.split('.')[0])]

# extract the frames from validation video
validation_video= video_list.pop(random.randint(0,len(video_list)-1))  # get as validation video the last video

val_dataset  = [extract_frames(video_path+validation_video),
                DataAudioExtraction(video_path+validation_video,anno_file=annotation_path,info_file=info_path),
                getAnnotations(annotation_path, validation_video.split('.')[0])]

print("Test Video:",test_video
      ,"\nValidation Video:",validation_video)


def videoSumm(annotation_path=None, info_path=None, video_path=None, summary_video_path=None,video_list=None):
    for video in video_list: 

        if(os.path.exists(f'{summary_video_path}{video}')):
            continue

        print("VIDEO:",video)

        # Extract frames 1/1 from the video
        original_frames=extract_frames(video_path+video, frame_rate=1)
        sample_frames=extract_frames(video_path+video)

        # Extract Data
        audio_features = DataAudioExtraction(video_path+video,anno_file=annotation_path,info_file=info_path)
        # ONLY AUDIO FEATURES HERE
        
        labels = getAnnotations(annotation_path, video.split('.')[0])
        
        
        importance=callNN(sample_frames,audio_features,labels=labels,test_dataset=test_dataset,val_dataset=val_dataset)
        
        # Maping to 1/1 rate
        importance=map_scores_to_original_frames(importance, 15)

        # get the best cluster
        clip_info = getClipImportances(importance,video)
        # Extracting values (importance scores) and weights (number of frames)
        values = [score for score, frames in clip_info.values()]
        weights = [frames for score, frames in clip_info.values()]

        # print("Values:",values)
        # print("Weights:",weights)
        # Calculate the total number of frames in the video
        total_frames = len(original_frames)
        # print("Total Frames:",total_frames)

        # Calculate the capacity as 15% of the total number of frames
        capacity = int(0.16 * total_frames)
        # print("Capacity:",capacity)

        # Now apply the knapsack algorithm
        selected_clips = knapsack_for_video_summary(values, weights, capacity)
        # print("Summary Indices:",selected_clips)
        
        selected_indices=getSelectedIndicesFromClips(selected_clips,video)
        # print('Sum Len Frame:',len(selected_indices))
        
        summary_frames=[original_frames[i] for i in selected_indices]
        
        binary_summary_frames=np.zeros(len(original_frames))
        binary_summary_frames[selected_indices]=1
        
        # Evaluate
        evaluation_method(ground_truth_path, binary_summary_frames, video.split('.')[0])
        
        # print(tabulate([evaluated_metrics], headers=['Threshold', 'Precision', 'Recall', 'F1', 'Avg. Importance', 'Max. Importance', 'Prop. High Importance']))
            
        video_name=video.split('.')[0]
        
        if(not os.path.exists(f'video_ext_data/{video_name}')):
            os.makedirs(f'video_ext_data/{video_name}')
        
        # Create Summary Video
        create_video_from_frames(summary_frames,f"{summary_video_path}{video}" , 30)
        
        # Extract data from video the next video
        getDataFlag=False
        deletePKLfiles(video_name)
        
    # return
    
videoSumm(annotation_path, info_path, video_path, summary_video_path, video_list)