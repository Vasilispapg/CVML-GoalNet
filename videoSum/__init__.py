
import sys
sys.path.append('DataExtraction')
sys.path.append('Evaluation')
sys.path.append('videoSum')
sys.path.append('knapsack')
sys.path.append('model')
import os
import numpy as np
from parseArgs import get_arguments

from frames import extract_visual_frames
from mapping import map_scores_to_original_frames
from clipImportance import getClipImportances
from clipImportance import getSelectedIndicesFromClips
from knapsack import knapsack_for_video_summary
from fscoreEval import evaluation_method
from deletePkl import deletePKLfiles
from videoCreator import create_video_from_frames
from data import DataExtraction
from labels import get_annotations
import random
from train import train
from pred import pred


def proccess(original_frames_train, video, importance):
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
        total_frames = len(original_frames_train)
        # print("Total Frames:",total_frames)

        # Calculate the capacity as 15% of the total number of frames
        capacity = int(0.16 * total_frames)
        # print("Capacity:",capacity)

        # Now apply the knapsack algorithm
        selected_clips = knapsack_for_video_summary(values, weights, capacity)
        # print("Summary Indices:",selected_clips)
        
        selected_indices=getSelectedIndicesFromClips(selected_clips,video)
        # print('Sum Len Frame:',len(selected_indices))
        
        summary_frames=[original_frames_train[i] for i in selected_indices]
        
        binary_summary_frames=np.zeros(len(original_frames_train))
        binary_summary_frames[selected_indices]=1
        
        # Evaluate
        evaluation_method(ground_truth_path, binary_summary_frames, video.split('.')[0])
        
        # print(tabulate([evaluated_metrics], headers=['Threshold', 'Precision', 'Recall', 'F1', 'Avg. Importance', 'Max. Importance', 'Prop. High Importance']))
            
        video_name=video.split('.')[0]
        
        if(not os.path.exists(f'video_ext_data/{video_name}')):
            os.makedirs(f'video_ext_data/{video_name}')
        
        # Create Summary Video
        create_video_from_frames(summary_frames,f"{summary_video_path}{video}" , 30)
        
        deletePKLfiles(video_name)

if __name__ == '__main__':
    
    annotation_path='datasets/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv' # Path for importantce (ground truth (partial))
    info_path='datasets/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv' # Path for the videos (e.g. links)
    video_path='datasets/ydata-tvsum50-v1_1/video/' # Input videos
    summary_video_path='datasets/summary_videos/' # Output pou vazoume ta video summaries
    ground_truth_path='datasets/ydata-tvsum50-v1_1/ground_truth/ydata-tvsum50.mat' # Katigoria annotation users etc (ground truth partial)
    video_list = [video for video in os.listdir(video_path) if video.endswith('.mp4')]  # List comprehension


    args=get_arguments()
    if args.train:
        print("Training")
        for video in video_list: 

            if(os.path.exists(f'{summary_video_path}{video}')):
                continue
            
            
            # extract the frames from validation video
            validation_video= video_list.pop(random.randint(0,len(video_list)-1))  # get as validation video the last video

            val_dataset  = [extract_visual_frames(video_path+validation_video),
                            DataExtraction(video_path+validation_video),
                            get_annotations(annotation_path, validation_video.split('.')[0])]
            print("Validation Video:",validation_video)
            

            print("VIDEO:",video)

            # Extract frames 1/1 from the video
            original_frames_train=extract_visual_frames(video_path+video, skip_frames=1)
            sample_frames_train=extract_visual_frames(video_path+video)

            # Extract Data
            audio_features_train = DataExtraction(video_path+video)
            # ONLY AUDIO FEATURES HERE
            
            labels_train_video = get_annotations(annotation_path, video.split('.')[0])
            
            importance=train([sample_frames_train, val_dataset[0]],
                            [audio_features_train, val_dataset[1]],
                            [labels_train_video, val_dataset[2]])      
            
            proccess(original_frames_train, video, importance)    
    
    elif args.pred:
        print("Predicting")
        for video in video_list: 

            if(os.path.exists(f'{summary_video_path}{video}')):
                continue
            

            print("VIDEO:",video)

            # Extract frames 1/1 from the video
            original_frames_train=extract_visual_frames(video_path+video, skip_frames=1)
            sample_frames_train=extract_visual_frames(video_path+video)

            # Extract Data
            audio_features_train = DataExtraction(video_path+video)
            # ONLY AUDIO FEATURES HERE
            
            labels_train_video = get_annotations(annotation_path, video.split('.')[0])
            
            importance=pred()  
            
            proccess(original_frames_train, video, importance)
    else:
        print("No action")
        pass

