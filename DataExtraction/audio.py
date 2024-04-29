
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
from scipy.interpolate import interp1d

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
    
    
def interpolate_mfcc(mfccs_current_frame, T):
    """
    The function `interpolate_mfcc` takes a set of MFCC coefficients for a current frame and
    interpolates them to a new length T using cubic interpolation.
    
    :param mfccs_current_frame: MFCCs (Mel-frequency cepstral coefficients) for the current frame of
    audio data. It is a 2D array where each row represents a different MFCC feature and each column
    represents a time step
    :param T: The parameter `T` in the `interpolate_mfcc` function represents the number of desired
    frames for interpolation. It determines how many frames you want to interpolate between the original
    MFCC frames in `mfccs_current_frame`. Increasing `T` will result in a smoother interpolation with
    more frames,
    :return: The function `interpolate_mfcc` returns the interpolated MFCCs (Mel-frequency cepstral
    coefficients) for the current frame with a new length specified by `T`.
    """
    original_indices = np.arange(mfccs_current_frame.shape[1])
    new_indices = np.linspace(0, mfccs_current_frame.shape[1] - 1, T)
    interpolated_mfccs = np.zeros((mfccs_current_frame.shape[0], T))

    for i in range(mfccs_current_frame.shape[0]):
        interpolator = interp1d(original_indices, mfccs_current_frame[i, :], kind='cubic')
        interpolated_mfccs[i, :] = interpolator(new_indices)

    return interpolated_mfccs
    
    
def extract_audio_features(audio_fp: str, n_frames: int) -> np.array:
    '''
        Returns:
            mfccs_per_frame. Shape (N,F,T), where N is the number of frames, F is the number of MFC coefficients and T is the frame's bin time length.
    '''

    T = 26
    y, sr = librosa.load(audio_fp)
    audio_samples_per_frame = len(y) / n_frames
    mfccs_per_frame = []
    for frame in range(n_frames):
        start_sample = round(frame * audio_samples_per_frame)
        end_sample = round(start_sample + audio_samples_per_frame)
        
        # Ensure we don't go beyond the audio length
        if end_sample > len(y):
            end_sample = len(y)

        mfccs_current_frame = librosa.feature.mfcc(y=y[start_sample:end_sample], sr=sr, n_mfcc=30)
        # Interpolation across the time axis for a given bin
        
        mfccs_currect_frame_interpolated = []
        for f_idx in range(mfccs_current_frame.shape[0]):
            interpolator = interp1d(
                np.arange(mfccs_current_frame.shape[1]), 
                mfccs_current_frame[f_idx, :], 
                kind='cubic',
                fill_value="extrapolate"  # This handles extrapolation
            )
            mfccs_currect_frame_interpolated.append(interpolator(np.linspace(0, mfccs_current_frame.shape[1] - 1, T)))
        mfccs_currect_frame_interpolated = np.array(mfccs_currect_frame_interpolated)
        assert mfccs_currect_frame_interpolated.shape[-1] == T, 'E: Shape mismatch'
        mfccs_per_frame.append(mfccs_currect_frame_interpolated)
    mfccs_per_frame = np.array(mfccs_per_frame)

    return mfccs_per_frame