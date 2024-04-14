
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
    M = 26

    # Load audio with a specific sampling rate
    y, sr = librosa.load(audio_output_path)

    # Calculate samples per frame based on audio length and video frame count
    audio_samples_per_frame = len(y) / num_frames

    mfccs_per_frame = []

    for frame in range(num_frames):
        start_sample = round(frame * audio_samples_per_frame)
        end_sample = round(start_sample + audio_samples_per_frame)

        # Extract MFCCs for the segment
        mfccs_current_frame = librosa.feature.mfcc(y=y[start_sample:end_sample], sr=sr, n_mfcc=30)
        DRAFT = mfccs_current_frame.shape[-1]
        # Correct shape
        if mfccs_current_frame.shape[-1] < M:
            # Pad with repetition
            mfccs_current_frame = np.repeat(mfccs_current_frame, np.ceil(M / mfccs_current_frame.shape[-1]), axis=1)[:, :M]
        else:
            # Crop
            mfccs_current_frame = mfccs_current_frame[:, :M]

        # shape : (30,26)

        mfccs_per_frame.append(mfccs_current_frame)
    assert mfccs_current_frame.shape[-1] == 26, 'E: Shape mismatch'

    return mfccs_per_frame