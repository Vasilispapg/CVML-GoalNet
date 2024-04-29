from frames import extract_frames
from audio import extract_audio_from_video, extract_audio_features_for_each_frame
from save import saveData
from getData import getData
from frames import extract_frames


def extractData(video_path, flag_to_extract):
    
    return_data=[]
    # Extract frames from the video
    if(flag_to_extract[0]):
        frames = extract_frames(video_path)
        return_data.append(['frames',frames])
    else:
        return_data.append(None)

    # Extract audio
    if(flag_to_extract[1]):
        audio_output_path = 'datasets/extractedAudio/extracted_audio.wav'
        extract_audio_from_video(video_path, audio_output_path) 

        # Extract audio features
        audio_features = extract_audio_features_for_each_frame(audio_output_path=audio_output_path,num_frames=len(frames))
        return_data.append(['audio',audio_features])
    else:
        return_data.append(None)

    return return_data


def DataAudioExtraction(video_path, info_file):
    video=video_path.split('/')[-1].split('.')[0] # file name, bc video_path has the extension as well (namely `filename.extension`)
    
    audio_features=getData('audio',video) # MFCC features. list[np.ndarray[np.float32]]. Shape (461, 128).
        
    if(audio_features is None):
        # Extract data from video and save it
        data=extractData(video_path, info_file,[True,True])
        # Save extracted Data
        for d in data:
            if d is not None:
                if(d[0]=='frames'):
                    frames=d[1]
                elif(d[0]=='audio'):
                    audio_features=d[1]
                saveData(d[0],d[1],video)
                
    return audio_features

