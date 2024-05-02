import torch
import os
import sys
import torch
import glob
import torch.optim as optim
from time import time
import h5py
import torch.nn as nn
import numpy as np

from utils import get_annotations, extract_condensed_frame_tensor, export_audio_from_video, extract_audio_features, dataloader, audio_visual_model, expand_array, get_video_data_from_h5, get_video_data_from_mat, get_clip_information, knapsack, export_video, get_frame_tensor

def train_importance_model():

    ## ! Initialization: Begin

    # Paths
    annotation_fp = 'ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv'
    audio_fp = './tmp/audio.wav'
    video_fps = sorted(glob.glob('./ydata-tvsum50-v1_1/video/*.mp4'), reverse = False)

    # Hyperparameters (preprocessing)
    videos_limitation = 3 # 999_999
    skip_frames = 15

    # Hyperparameters (training process - frame importance model)
    num_epochs = 1
    lr = 0.00008

    ## ! Initialization: End

    video_fps = video_fps[:videos_limitation]

    ground_truths_trimmed = []
    ground_truths_full = []
    for video_fp in video_fps:
        video_id = video_fp.split('/')[-1].split('.')[0]
        ground_truth_trimmed, ground_truth_full = get_annotations(annotation_fp = annotation_fp, video_id = video_id, skip_frames = skip_frames)
        ground_truths_trimmed.append(ground_truth_trimmed)
        ground_truths_full.append(ground_truth_full)
    ground_truths_trimmed = ground_truths_trimmed[:videos_limitation]
    ground_truths_full = ground_truths_full[:videos_limitation]
    full_n_frames = len(ground_truth_full)

    frames = []
    audios = []
    for video_fp in video_fps:
        print("Extracting content from %s"%(video_fp))
        visual_frames_tensor = extract_condensed_frame_tensor(video_fp, skip_frames = skip_frames)
        N = len(visual_frames_tensor)
        export_audio_from_video(audio_fp = audio_fp, video_fp = video_fp)
        audio_features_tensor = extract_audio_features(audio_fp = audio_fp, n_frames = N)
        frames.append(visual_frames_tensor)
        audios.append(audio_features_tensor)

    val_dataset = dataloader(frames = [frames.pop()], audios = [audios.pop()], labels = [ground_truths_trimmed.pop()])
    train_dataset = dataloader(frames = frames, audios = audios, labels = ground_truths_trimmed)
    del frames, audios, ground_truths_trimmed

    print("Number of train videos: %d"%(len(train_dataset)))
    print("Number of val videos: %d"%(len(val_dataset)))

    frame_importance_model = audio_visual_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params = frame_importance_model.parameters(), lr = lr)

    val_frames, val_audios, val_labels = val_dataset[0]

    # Evaluation of model prior to training
    print("\n[Initial model evaluation]")
    t0 = time()
    initial_train_losses = []
    for video_idx, (batch_frames, batch_audios, batch_labels) in enumerate(train_dataset):
        with torch.no_grad():
            initial_train_losses.append(criterion(frame_importance_model(batch_audios, batch_frames), batch_labels).item())
        torch.cuda.empty_cache()
    initial_train_loss = sum(initial_train_losses) / len(initial_train_losses)
    with torch.no_grad():
        initial_val_loss = criterion(frame_importance_model(val_audios, val_frames), batch_labels).item()
    t1 = time()
    print("Train loss: %.4f - Val loss: %.4f - Time: %.2fs"%(initial_train_loss, initial_val_loss, t1-t0))

    # Training
    print("\n[Training state]")
    optimal_est_train_loss = 9999
    opt_epoch = -1
    for epoch in range(num_epochs):
        batch_losses = []
        print("Epoch: %d/%d"%(epoch, num_epochs-1))
        for video_idx, (batch_frames, batch_audios, batch_labels) in enumerate(train_dataset):
            t0 = time()
            optimizer.zero_grad()
            batch_predictions = frame_importance_model(batch_audios, batch_frames)
            batch_loss = criterion(batch_predictions, batch_labels)
            batch_loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_loss = criterion(frame_importance_model(val_audios, val_frames), val_labels).item()
            batch_losses.append(batch_loss.item())
            t1 = time()
            print("Video: %d/%d - Batch size: %d - Batch loss: %.4f - Val loss: %.4f - Time: %.2fs"%(video_idx, len(train_dataset)-1, len(batch_frames), batch_loss.item(), val_loss, t1-t0))
            torch.cuda.empty_cache()
        train_est_loss = sum(batch_losses) / len(batch_losses)
        print("Est. train loss: %.4f - Val loss: %.4f"%(train_est_loss, val_loss))
        if train_est_loss < optimal_est_train_loss:
            optimal_est_train_loss = train_est_loss
            opt_epoch = epoch
        print()
    print("Optimal est. train loss: %.4f @ epoch %d"%(optimal_est_train_loss, opt_epoch))
    torch.save(obj = frame_importance_model.state_dict(), f = "./models/frame_importance_model.pt")

def infer(video_fp: str):

    audio_fp = './tmp/audio.wav'
    h5_file_path = 'ydata-tvsum50-v1_1/ground_truth/eccv16_dataset_tvsum_google_pool5.h5'
    mat_file_path = 'ydata-tvsum50-v1_1/ground_truth/ydata-tvsum50.mat'

    skip_frames = 15

    print("VIDEO:", video_fp)

    visual_frames_tensor = torch.tensor(extract_condensed_frame_tensor(video_fp, skip_frames = skip_frames), dtype = torch.float32)
    full_visual_frames_tensor = get_frame_tensor(video_fp)
    full_n_frames = len(full_visual_frames_tensor)

    export_audio_from_video(audio_fp = audio_fp, video_fp = video_fp)
    N = len(visual_frames_tensor)
    audio_features_tensor = torch.tensor(extract_audio_features(audio_fp = audio_fp, n_frames = N), dtype = torch.float32)

    data = dataloader(frames = [visual_frames_tensor], audios = [audio_features_tensor], labels = None)

    frame_importance_model = audio_visual_model()
    frame_importance_model.load_state_dict(torch.load(f = "./models/frame_importance_model.pt"))

    batch_frames, batch_audios, _ = next(iter(data))

    batch_predictions = frame_importance_model(batch_audios, batch_frames)
    batch_predictions = torch.round(batch_predictions[:, 0]).type(torch.int8).tolist()
    expanded_batch_predictions = expand_array(arr = batch_predictions, expansion_rate = skip_frames, length = full_n_frames)

    # ! Clips and Knapsack optimization: Begin

    video_id_ = video_fp.split('/')[-1].split('.')[0]

    video_data_h5 = get_video_data_from_h5(h5_file_path)
    video_data_mat = get_video_data_from_mat(mat_file_path)

    video_id_map = {}
    for video_mat in video_data_mat:
        for video_h5 in video_data_h5:
            if video_mat[1] == video_h5[1] + 1:
                video_id_map[video_mat[0]] = video_h5[0]

    with h5py.File(h5_file_path, 'r') as file:
        clip_intervals = file[video_id_map[video_id_]]['change_points'][:]

    clip_importances, clip_lengths = get_clip_information(clip_intervals = clip_intervals, importances = expanded_batch_predictions)

    max_knapsack_capacity = int(0.15 * full_n_frames)

    summarization_clip_interval_indices = knapsack(values = clip_importances, weights = clip_lengths, capacity = max_knapsack_capacity)
    summarization_clip_intervals = [clip_intervals[summarization_clip_interval_index] for summarization_clip_interval_index in summarization_clip_interval_indices]
    summarized_video = np.concatenate([full_visual_frames_tensor[summarization_clip_interval[0]:summarization_clip_interval[1]] for summarization_clip_interval in summarization_clip_intervals], axis=0)

    print("Exporting video %s"%(video_id_))
    export_video(frames = summarized_video, output_path = "./tmp/summarized_video.mp4", frame_rate = 30)

    # ! Clips and Knapsack optimization: End


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--train-importance-model':
        train_importance_model()
    elif len(sys.argv) == 2 and sys.argv[1] == '--infer':
        video_fp = 'ydata-tvsum50-v1_1/video/-esJrBWj2d8.mp4'
        infer(video_fp = video_fp)