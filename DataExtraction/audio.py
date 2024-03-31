
from moviepy.editor import VideoFileClip
import librosa
import numpy as np

def extract_audio_from_video(video_path, audio_output_path):
    """
    The function `extract_audio_from_video` extracts the audio from a video file and saves it to a
    specified output path.
    
    :param video_path: The `video_path` parameter is the file path to the video from which you want to
    extract the audio. This should be a valid path to the video file on your system
    :param audio_output_path: The `audio_output_path` parameter is the file path where you want to save
    the extracted audio from the video. This should include the file name and extension of the audio
    file you want to create
    """
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output_path)
    video.close()
    
def extract_audio_features_for_each_frame(audio_output_path: str, num_frames: int):
    """
    The function `extract_audio_features_for_each_frame` extracts MFCC features for each frame of an
    audio file based on a specified number of frames.
    
    :param audio_output_path: The `audio_output_path` parameter is a string that represents the file
    path to the audio file from which you want to extract audio features. This function loads the audio
    file using librosa and then extracts MFCC (Mel-frequency cepstral coefficients) features for each
    frame of the audio
    :type audio_output_path: str
    :param num_frames: The `num_frames` parameter represents the number of frames you want to divide the
    audio into for feature extraction. This value determines how many segments the audio will be divided
    into, with each segment processed to extract MFCCs (Mel-frequency cepstral coefficients) for
    analysis
    :type num_frames: int
    :return: The function `extract_audio_features_for_each_frame` returns a list of MFCCs (Mel-frequency
    cepstral coefficients) processed for each frame of the audio file.
    """
    
    # Load audio with a specific sampling rate
    y, sr = librosa.load(audio_output_path)

    # Calculate samples per frame based on audio length and video frame count
    audio_samples_per_frame = int(len(y) / num_frames)
    
    mfccs_per_frame = []

    for frame in range(num_frames):
        start_sample = frame * audio_samples_per_frame
        end_sample = start_sample + audio_samples_per_frame
        end_sample = min(end_sample, len(y))  # Ensure not exceeding audio length

        # Extract MFCCs for the segment
        mfccs_current_frame = librosa.feature.mfcc(y=y[start_sample:end_sample], sr=sr, n_mfcc=30)
        mfccs_processed = np.mean(mfccs_current_frame.T, axis=0)
        
        mfccs_per_frame.append(mfccs_processed)
    
    return mfccs_per_frame
