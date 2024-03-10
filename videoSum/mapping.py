import numpy as np
import h5py

def map_frames_to_labels_with_indices(frames, labels):
    label_frame_dict = {}
    for label, (frame_index, frame) in zip(labels, enumerate(frames)):
        # Convert label from numpy array to scalar if necessary
        label_scalar = label.item() if isinstance(label, np.ndarray) else label
        if label_scalar not in label_frame_dict:
            label_frame_dict[label_scalar] = []
        label_frame_dict[label_scalar].append((frame_index, frame))
    return label_frame_dict


# Get the last change point and total frames for each video in the .h5 file
def get_video_data_from_h5(file_path):
    video_data_h5 = []
    with h5py.File(file_path, 'r') as file:
        for video_id in file.keys():
            last_change_point = file[str(video_id)]['change_points'][-1]
            total_frames = last_change_point[1]
            video_data_h5.append([video_id, total_frames])
    return video_data_h5

# Get frame numbers from the .mat file
def get_frame_numbers(encoded_frames, hdf5_file):
    frame_numbers = []
    for ref_array in encoded_frames:
        for ref in ref_array:
            frame_data = hdf5_file[ref]
            frame_numbers.extend([int(char[0]) for char in frame_data])
    return frame_numbers

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

# Extract data from .mat file
def get_video_data_from_mat(file_path):
    video_data_mat = []
    breakpoint()
    with h5py.File(file_path, 'r') as f:
        encoded_videos = f['tvsum50']['video'][:]
        encoded_frame_counts = f['tvsum50']['nframes'][:]
        decoded_videos = decode_titles(encoded_videos, f)
        decoded_frame_counts = get_frame_numbers(encoded_frame_counts, f)
        for i, video_id in enumerate(decoded_videos):
            video_data_mat.append([video_id, decoded_frame_counts[i]])
    return video_data_mat

def getChangingPoints(video_id):
    with h5py.File(h5_file_path, 'r') as file:
        return file[video_id]['change_points'][:]
    
    
def map_scores_to_original_frames(sampled_scores, frame_rate):
    # Create an empty list to hold the mapped scores
    original_scores = []

    # Iterate over the sampled scores
    for score in sampled_scores:
        # Replicate each score frame_rate times
        original_scores.extend([score] * frame_rate)

    return original_scores

# Comparing and mapping the data
h5_file_path = 'datasets/ydata-tvsum50-v1_1/eccv16_dataset_tvsum_google_pool5.h5'
mat_file_path = 'datasets/ydata-tvsum50-v1_1/ground_truth/ydata-tvsum50.mat'

video_data_h5 = get_video_data_from_h5(h5_file_path)
video_data_mat = get_video_data_from_mat(mat_file_path)

video_id_map = {}
for video_mat in video_data_mat:
    for video_h5 in video_data_h5:
        if video_mat[1] == video_h5[1] + 1:
            video_id_map[video_mat[0]] = video_h5[0]