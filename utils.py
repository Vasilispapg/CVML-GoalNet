import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import csv
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from moviepy.editor import VideoFileClip
import numpy as np
import librosa
import h5py
import pandas as pd


class dataloader(Dataset):

    def __init__(self, fps: list[str], frames: list[torch.tensor], full_n_frames: list[np.ndarray], audios: list[torch.tensor], labels: list[torch.tensor] = None, device: str = 'cpu'):
        '''
            Description
                Video summarization dataset. Each batch is one individual video. Each instance is one frame belonging to the mentioned video-batches.

            Parameters
                fps. File path of a given instance video.
                frames. For a given video index i, frames[i] has shape (N_i, C, H, W) and each frame is preprocessed.
                full_n_frames. Total number of frames from raw videos.
                audios. For a given video index i, audios[i] has shape (N_i, K, n_mfcc, B).
                labels. For a given video index i, labels has shape (N_i,).
                device. Processing unit identifier, responsible for tensor operations.

                N is the number of frames; B is the number of timestamps per bin of the MFCC algorithm; C is the number of visual channels.
        '''

        self.fps = fps
        self.video_ids = [video_fp.split('/')[-1].split('.')[0] for video_fp in self.fps]
        self.frames = [torch.tensor(frames_, dtype = torch.float32, device = device) for frames_ in frames]
        self.full_n_frames = full_n_frames
        self.audios = [torch.tensor(audios_, dtype = torch.float32, device = device) for audios_ in audios]
        if labels is None:
            self.labels = [None for _ in range(len(frames))]
            assert len(self.frames) == len(self.audios), 'E: Inconsistency in data loader definition'
        else:
            self.labels = [torch.tensor(labels_, dtype = torch.float32, device = device) for labels_ in labels]
            assert len(self.frames) == len(self.audios) == len(self.labels), 'E: Inconsistency in data loader definition'

        self.titles = self.get_titles(self.video_ids)

        self.N = len(self.frames)

    def get_titles(self, video_ids):

        df = pd.read_csv("./ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv", sep = "\t")
        titles_unordered = dict()
        for unordered_idx, row in df.iterrows():
            titles_unordered[row["video_id"]] = row["title"]

        titles_ordered = []
        for video_id in video_ids:
            titles_ordered.append(titles_unordered[video_id])

        return titles_ordered

    def __len__(self):
        return self.N

    def __getitem__(self, video_idx):

        self.title = self.titles[video_idx]
        self.full_n_frames_ = self.full_n_frames[video_idx]

        return self.video_ids[video_idx], self.frames[video_idx], self.audios[video_idx], self.labels[video_idx]

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
class Cnn0(nn.Module):

    def __init__(self):
        super(Cnn0, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32,64,5,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = SeparableConv2d(64,128,5,2,1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(128,256,3,2,1)
        self.bn4 = nn.BatchNorm2d(256)

    def features(self, input):

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def forward(self, input):
        x = self.features(input)
        return x

class audio_visual_model(nn.Module):

    def __init__(self, audio_included):

        super(audio_visual_model, self).__init__()

        self.audio_included = audio_included

        self.visual_model = Cnn0()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Audio Branch (Simple CNN for MFCC); the sliding window covers all the Mel coefficients, for some fixed time units totalling <kernel_size>, and slides across the time unit axis; therefore for a given sliding window, the output considers all Mel coefficients for these particular fixed time units and returns one value
        self.audio_model = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid()
        )

    def forward(self, audio_input, visual_input):


        with torch.no_grad():
            features = self.visual_model(visual_input)  # Extract features using Xception
        visual_features = nn.ReLU(inplace=True)(features)
        visual_features = F.adaptive_avg_pool2d(visual_features, (1, 1))
        visual_features = visual_features.view(visual_features.size(0), -1)
        visual_features = nn.LazyLinear(512).to(visual_features.device)(visual_features)

        if self.audio_included:
            audio_features = self.audio_model(audio_input)
            combined_features = torch.cat((audio_features, visual_features), axis = -1)
        else:
            combined_features = visual_features

        output = 4 * self.fusion(combined_features) + 1
        return output

def extract_condensed_frame_tensor(fp: str, skip_frames: int):

    video = cv2.VideoCapture(fp)
    count = 0
    success = True
    frames = []

    while success:
        success, image = video.read()
        if count % skip_frames == 0 and success:
            image = ((image - image.min()) / (image.max() - image.min() + 1e-7)).astype(np.float32)
            image = cv2.resize(image, (50, 50))
            frames.append(image)
        count += 1
    full_n_frames = count-1

    video.release()

    return np.transpose(np.array(frames), axes = (0, 3, 1, 2)), full_n_frames

def get_frame_tensor(fp: str):

    video = cv2.VideoCapture(fp)
    success = True
    frames = []

    while success:
        success, image = video.read()
        frames.append(image)
    frames = frames[:-1]

    return np.array(frames)

def export_audio_from_video(audio_fp, video_fp):
    video = VideoFileClip(video_fp)
    audio = video.audio
    audio.write_audiofile(audio_fp)
    video.close()

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

def get_visual_features(model, sample_visual_frames):
    model.eval()
    visual_descriptor_tensor = []
    for frame in sample_visual_frames:
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        output = model(frame)
        visual_descriptor_tensor.append(output)

    return visual_descriptor_tensor

# Get frame numbers from the .mat file
def get_frame_numbers(encoded_frames, hdf5_file):
    frame_numbers = []
    for ref_array in encoded_frames:
        for ref in ref_array:
            frame_data = hdf5_file[ref]
            frame_numbers.extend([int(char[0]) for char in frame_data])
    return frame_numbers

def get_annotations(annotation_fp, video_id, skip_frames):
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
    mean_annotations_trimmed=[]
    mean_annotations_full = []
    count=0
    for ann in annotations:
        mean_ann = np.mean(ann)
        if (count % skip_frames) == 0:
            mean_annotations_trimmed.append(mean_ann)
        mean_annotations_full.append(mean_ann)
        count+=1
    mean_annotations_trimmed = np.array(mean_annotations_trimmed)

    return mean_annotations_trimmed, mean_annotations_full

def expand_array(arr, expansion_rate, length):
    expanded_arr = []
    for el in arr:
        expanded_arr += [el] * expansion_rate

    if len(expanded_arr) >= length:
        expanded_arr = expanded_arr[:length]
    else:
        expanded_arr += [expanded_arr[-1]] * (length - len(expanded_arr))

    return expanded_arr

def decode_titles(encoded_titles, hdf5_file):
    decoded_titles = []
    for ref_array in encoded_titles:
        # Handle the case where each ref_array might contain multiple references
        for ref in ref_array:
            # Dereference each HDF5 object reference to get the actual data
            title_data = hdf5_file[ref]
            # Decode the title
            decoded_title = ''.join(chr(char[0]) for char in title_data)
            decoded_titles.append(decoded_title)
    return decoded_titles

def get_video_data_from_h5(file_path):
    video_data_h5 = []
    with h5py.File(file_path, 'r') as file:
        for video_id in file.keys():
            last_change_point = file[str(video_id)]['change_points'][-1]
            total_frames = last_change_point[1]
            video_data_h5.append([video_id, total_frames])
    return video_data_h5

def get_video_data_from_mat(file_path):
    video_data_mat = []
    with h5py.File(file_path, 'r') as f:
        encoded_videos = f['tvsum50']['video'][:]
        encoded_frame_counts = f['tvsum50']['nframes'][:]
        decoded_videos = decode_titles(encoded_videos, f)
        decoded_frame_counts = get_frame_numbers(encoded_frame_counts, f)
        for i, video_id in enumerate(decoded_videos):
            video_data_mat.append([video_id, decoded_frame_counts[i]])
    return video_data_mat


def get_clip_information(clip_intervals: list[list[int]], importances: list[int]) -> tuple[list]:
    '''
        Parameters:
            clip_intervals. Contains the intervals of each clip from a temporally segmented video. Each interval is expressed as a list containing two integers, where the former integer is the initial index of the video, and the latter integer is the final index of the video. The indexing function is defined on the corresponding full video.
            importances. Contains the importance of each frame from a full video.

        Returns:
            clip_importance. Each item holds the importance of the corresponding clip.
            clip_length. Each item holds the length of the corresponding clip.
    '''

    full_n_frames = len(importances)
    assert clip_intervals[-1][-1] + 1 == full_n_frames, "E: Incompatible lengths"
    clip_importances = []
    clip_lengths = []
    for clip_interval in clip_intervals:
        importances_slice = importances[clip_interval[0]:clip_interval[1]]
        clip_importances.append(sum(importances_slice))
        clip_lengths.append(len(importances_slice))

    return clip_importances, clip_lengths

def knapsack(values, weights, capacity, scale_factor=5):
    """
    Apply the 0/1 Knapsack algorithm to select video segments for summarization.

    :param values: List of importance scores for each segment.
    :param weights: List of durations for each segment in seconds.
    :param capacity: Maximum total duration for the summary in seconds.
    :param scale_factor: Factor to scale weights to integers.
    :return: Indices of the segments to include in the summary.
    """

    # Scale weights and capacity
    weights = [int(w * scale_factor) for w in weights]
    capacity = int(capacity * scale_factor)

    n = len(values)
    K = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Build table K[][] in a bottom-up manner
    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif weights[i-1] <= w:
                K[i][w] = max(values[i-1] + K[i-1][w-weights[i-1]], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]

    # Find the selected segments
    res = K[n][capacity]
    w = capacity
    selected_indices = []

    for i in range(n, 0, -1):
        if res <= 0:
            break
        if res == K[i-1][w]:
            continue
        else:
            selected_indices.append(i-1)
            res = res - values[i-1]
            w = w - weights[i-1]

    selected_indices.reverse()
    return selected_indices

def export_video(frames, output_path, frame_rate=30):

    height, width, _ = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    
    for frame in frames:
        out.write(frame)

    out.release()

def load_mat_file(file_path,videoID):
    """
    Load a .mat file and return its contents.

    :param file_path: Path to the .mat file.
    :return: Contents of the .mat file.
    """
    with h5py.File(file_path, 'r') as file:
        user_anno_refs=file['tvsum50']['user_anno'][:] # type: ignore
        video_refs=file['tvsum50']['video'][:] # type: ignore

        decoded_videos = decode_titles(video_refs,file)
    
        annotations = []        
        # Get the index from decoded video list to find the annotation for the video
        index = [i for i, x in enumerate(decoded_videos) if x.lower() in videoID.lower()][0]
        
        # Iterate over each reference
        for ref in user_anno_refs:
            # Dereference each HDF5 object reference
            ref_data = file[ref[0]]

            # Convert to NumPy array and add to the annotations list
            annotations.append(np.array(ref_data))
            
        return annotations[index]

def evaluate_summary(predicted_summary, user_summary, eval_method='avg'):
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        
        precision = sum(overlapped) / sum(S) if sum(S) != 0 else 0
        recall = sum(overlapped) / sum(G) if sum(G) != 0 else 0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        f_scores.append(f_score * 100)  # multiplied by 100 for percentage

    if eval_method == 'max':
        return max(f_scores)
    else:  # 'avg'
        return sum(f_scores) / len(f_scores)

def evaluation_method(ground_truth_path, summary_indices,video_id):

    # Get the ground_truth
    ground_truth = np.array(load_mat_file(ground_truth_path, video_id))

    f_score_max = evaluate_summary(summary_indices, ground_truth, 'max')
    f_score_avg = evaluate_summary(summary_indices, ground_truth)

    return f_score_avg, f_score_max

def postprocessing(video_id, h5_file_path, mat_file_path, batch_predictions, skip_frames, full_n_frames, full_frames):

    batch_predictions = torch.round(batch_predictions[:, 0]).type(torch.int8).tolist()

    expanded_batch_predictions = expand_array(arr = batch_predictions, expansion_rate = skip_frames, length = full_n_frames)

    video_data_h5 = get_video_data_from_h5(h5_file_path)
    video_data_mat = get_video_data_from_mat(mat_file_path)

    video_id_map = {}
    for video_mat in video_data_mat:
        for video_h5 in video_data_h5:
            if video_mat[1] == video_h5[1] + 1:
                video_id_map[video_mat[0]] = video_h5[0]

    with h5py.File(h5_file_path, 'r') as file:
        clip_intervals = file[video_id_map[video_id]]['change_points'][:]

    clip_importances, clip_lengths = get_clip_information(clip_intervals = clip_intervals, importances = expanded_batch_predictions)

    max_knapsack_capacity = int(0.15 * full_n_frames)

    summarization_clip_interval_indices = knapsack(values = clip_importances, weights = clip_lengths, capacity = max_knapsack_capacity)
    summarization_clip_intervals = [clip_intervals[summarization_clip_interval_index] for summarization_clip_interval_index in summarization_clip_interval_indices]
    summarized_video = np.concatenate([full_frames[summarization_clip_interval[0]:summarization_clip_interval[1]] for summarization_clip_interval in summarization_clip_intervals], axis=0)

    summarized_video_frame_indices = np.zeros(shape = (full_n_frames,), dtype = np.uint8)
    for summarization_clip_interval in summarization_clip_intervals:
        for frame_idx in range(summarization_clip_interval[0], summarization_clip_interval[1] + 1):
            summarized_video_frame_indices[frame_idx] = 1

    return summarized_video, summarized_video_frame_indices