import torch
import os
import sys
import torch
import glob
from time import time
import torch.nn as nn
import numpy as np
from utils import postprocess_and_get_fscores, get_dataloaders, AVM


if __name__ == '__main__':

    audio_included = False

    # Paths
    if audio_included:
        opt_frame_importance_model_fp = "./models/opt_frame_importance_model.pt"
        ckp_frame_importance_model_fp = "./models/ckp_frame_importance_model.pt"
        exported_image_fp = "./tmp/train_states.png"
    else:
        opt_frame_importance_model_fp = "./models/opt_frame_importance_model_no_audio.pt"
        ckp_frame_importance_model_fp = "./models/ckp_frame_importance_model_no_audio.pt"
        exported_image_fp = "./tmp/train_states_no_audio.png"
    annotation_fp = 'ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv'
    h5_file_path = 'ydata-tvsum50-v1_1/ground_truth/eccv16_dataset_tvsum_google_pool5.h5'
    mat_file_path = 'ydata-tvsum50-v1_1/ground_truth/ydata-tvsum50.mat'
    video_fps = sorted(glob.glob('./ydata-tvsum50-v1_1/video/*.mp4'), reverse = False)
    video_fps = ['./ydata-tvsum50-v1_1/video/37rzWOQsNIw.mp4', './ydata-tvsum50-v1_1/video/RBCABdttQmI.mp4']

    # Hyperparameters (preprocessing)
    skip_frames = 30

    # Hyperparameters (training process - frame importance model)
    train_ratio = 0.8
    np.random.seed(seed = 12344321)

    ## ! Initialization: End

    criterion = nn.MSELoss()

    np.random.shuffle(video_fps)

    train_dataset, val_dataset = get_dataloaders(video_fps = video_fps, skip_frames = skip_frames, train_ratio = train_ratio, annotation_fp = annotation_fp, mat_file_path = mat_file_path, h5_file_path = h5_file_path, audio_included = audio_included)

    print("Number of train videos: %d"%(len(train_dataset)))
    print("Number of val videos: %d"%(len(val_dataset)))

    est_train_losses = []
    est_train_f_scores_avg = []
    est_train_f_scores_max = []
    val_losses = []
    val_f_scores_avg = []
    val_f_scores_max = []

    s_l = 10 # sample length

    print("\n[Baseline computation]\n")

    with torch.no_grad():
        for s in range(s_l):

            frame_importance_model = AVM(audio_included = audio_included)

            train_losses_step = []
            train_f_scores_avg_step = []
            train_f_scores_max_step = []

            for video_idx, (video_id, batch_frames, batch_audios, batch_labels, batch_gd_summarized_video_frame_indices_) in enumerate(train_dataset):

                batch_predictions = frame_importance_model(batch_audios, batch_frames)
                # train_loss = criterion(batch_predictions, (batch_labels-1).long()).item()
                # batch_predictions = torch.argmax(batch_predictions, axis = 1) + 1
                train_loss = criterion(batch_predictions, batch_labels).item()
                full_n_batch_frames = train_dataset.full_n_frames_
                batch_f_score_avg, batch_f_score_max = postprocess_and_get_fscores(video_id = video_id, batch_predictions = batch_predictions, full_n_batch_frames = full_n_batch_frames, gd_summarized_video_frame_indices = batch_gd_summarized_video_frame_indices_, h5_file_path = h5_file_path, mat_file_path = mat_file_path, skip_frames = skip_frames)
                print("Video: %d/%d - ID: %s\nBatch Set - Size: %d - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.2f"%(video_idx, len(train_dataset)-1, video_id, full_n_batch_frames, train_loss, batch_f_score_avg, batch_f_score_max))
                train_losses_step.append(train_loss)
                train_f_scores_avg_step.append(batch_f_score_avg)
                train_f_scores_max_step.append(batch_f_score_max)

            val_losses_step = []
            val_f_scores_avg_step = []
            val_f_scores_max_step = []

            for video_id, val_frames, val_audios, val_labels, val_gd_summarized_video_frame_indices_ in val_dataset:

                val_predictions = frame_importance_model(val_audios, val_frames)
                # val_loss = criterion(val_predictions, (val_labels-1).long()).item()
                # val_predictions = torch.argmax(val_predictions, axis = 1) + 1
                val_loss = criterion(val_predictions, val_labels).item()
                full_n_val_frames = val_dataset.full_n_frames_
                val_f_score_avg, val_f_score_max = postprocess_and_get_fscores(video_id = video_id, batch_predictions = val_predictions, full_n_batch_frames = full_n_val_frames, gd_summarized_video_frame_indices = val_gd_summarized_video_frame_indices_, h5_file_path = h5_file_path, mat_file_path = mat_file_path, skip_frames = skip_frames)
                val_losses_step.append(val_loss)
                val_f_scores_avg_step.append(val_f_score_avg)
                val_f_scores_max_step.append(val_f_score_max)

            est_train_loss = sum(train_losses_step) / len(train_losses_step)
            est_train_f_score_avg = sum(train_f_scores_avg_step) / len(train_f_scores_avg_step)
            est_train_f_score_max = sum(train_f_scores_max_step) / len(train_f_scores_max_step)

            val_loss = sum(val_losses_step) / len(val_losses_step)
            val_f_score_avg = sum(val_f_scores_avg_step) / len(val_f_scores_avg_step)
            val_f_score_max = sum(val_f_scores_max_step) / len(val_f_scores_max_step)

            est_train_losses.append(est_train_loss)
            est_train_f_scores_avg.append(est_train_f_score_avg)
            est_train_f_scores_max.append(est_train_f_score_max)
            val_losses.append(val_loss)
            val_f_scores_avg.append(val_f_score_avg)
            val_f_scores_max.append(val_f_score_max)

            torch.cuda.empty_cache()

    est_train_losses_mean = sum(est_train_losses) / len(est_train_losses)
    est_train_f_scores_avg_mean = sum(est_train_f_scores_avg) / len(est_train_f_scores_avg)
    est_train_f_scores_max_mean = sum(est_train_f_scores_max) / len(est_train_f_scores_max)

    est_train_losses_opt = min(est_train_losses)
    est_train_f_scores_avg_opt = max(est_train_f_scores_avg)
    est_train_f_scores_max_opt = max(est_train_f_scores_max)

    val_losses_mean = sum(val_losses) / len(val_losses)
    val_f_scores_avg_mean = sum(val_f_scores_avg) / len(val_f_scores_avg)
    val_f_scores_max_mean = sum(val_f_scores_max) / len(val_f_scores_max)

    val_losses_opt = min(val_losses)
    val_f_scores_avg_opt = max(val_f_scores_avg)
    val_f_scores_max_opt = max(val_f_scores_max)

    print("Train Opt - Loss: %.4f - F-score Avg: %.2f - F-score Max: %.2f"%(est_train_losses_opt, est_train_f_scores_avg_opt, est_train_f_scores_max_opt))
    print("Val Opt - Loss: %.4f - F-score Avg: %.2f - F-score Max: %.2f"%(val_losses_opt, val_f_scores_avg_opt, val_f_scores_max_opt))

    print("Train Mean - Loss: %.4f - F-score Avg: %.2f - F-score Max: %.2f"%(est_train_losses_mean, est_train_f_scores_avg_mean, est_train_f_scores_max_mean))
    print("Val Mean - Loss: %.4f - F-score Avg: %.2f - F-score Max: %.2f"%(val_losses_mean, val_f_scores_avg_mean, val_f_scores_max_mean))