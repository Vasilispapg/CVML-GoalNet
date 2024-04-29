from audio import export_audio_from_video,extract_audio_features
from save import saveData
from getData import getData
from frames import extract_visual_frames



def extractData(video_path, flag_to_extract):
    
    return_data=[]
    # Extract frames from the video
    if(flag_to_extract[0]):
        frames = extract_visual_frames(video_path)
        return_data.append(['frames',frames])
    else:
        return_data.append(None)

    # Extract audio
    if(flag_to_extract[1]):
        audio_output_path = 'datasets/extractedAudio/extracted_audio.wav'
        export_audio_from_video(audio_output_path,video_path) 
    #     export_audio_from_video(audio_fp = audio_fp, video_fp = video_fp)

        # Extract audio features
        audio_features = extract_audio_features(audio_fp=audio_output_path,n_frames=len(frames))

        return_data.append(['audio',audio_features])
    else:
        return_data.append(None)

    return return_data
    

def DataExtraction(video_path):
    video=video_path.split('/')[-1].split('.')[0] # file name, bc video_path has the extension as well (namely `filename.extension`)
    
    audio_features=getData('audio',video) # MFCC features. list[np.ndarray[np.float32]]. Shape (461, 128).
        
    if(audio_features is None):
        # Extract data from video and save it
        data=extractData(video_path,[True,True])
        # Save extracted Data
        for d in data:
            if d is not None:
                if(d[0]=='frames'):
                    frames=d[1]
                elif(d[0]=='audio'):
                    audio_features=d[1]
                saveData(d[0],d[1],video)
                
    return audio_features