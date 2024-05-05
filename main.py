import torch
import os
import sys
import torch
import glob
import torch.optim as optim
from time import time
import torch.nn as nn
import numpy as np

from utils import get_annotations, extract_condensed_frame_tensor, export_audio_from_video, extract_audio_features, dataloader, audio_visual_model, export_video, get_frame_tensor, evaluation_method, postprocessing
from visualization import generate_metric_plots

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def train_importance_model(audio_included, load_ckp):

    def get_train_f_scores():

        # Clips and Knapsack optimization
        _, summarized_video_frame_indices = postprocessing\
        (
            video_id = video_id,
            h5_file_path = h5_file_path,
            mat_file_path = mat_file_path,
            batch_predictions = batch_predictions,
            skip_frames = skip_frames,
            full_n_frames = full_n_batch_frames,
            full_frames = None
        )

        # Summarization evaluation
        train_f_score_avg, train_f_score_max = evaluation_method(ground_truth_path = mat_file_path, summary_indices = summarized_video_frame_indices, video_id = video_id)

        return train_f_score_avg, train_f_score_max

    def get_val_f_scores():

        # Clips and Knapsack optimization
        _, summarized_video_frame_indices = postprocessing\
        (
            video_id = video_id,
            h5_file_path = h5_file_path,
            mat_file_path = mat_file_path,
            batch_predictions = val_predictions,
            skip_frames = skip_frames,
            full_n_frames = full_n_val_frames,
            full_frames = None
        )

        # Summarization evaluation
        val_f_score_avg, val_f_score_max = evaluation_method(ground_truth_path = mat_file_path, summary_indices = summarized_video_frame_indices, video_id = video_id)

        return val_f_score_avg, val_f_score_max

    ## ! Initialization: Begin

    # Paths
    if audio_included:
        opt_frame_importance_model_fp = "./models/opt_frame_importance_model.pt"
        ckp_frame_importance_model_fp = "./models/ckp_frame_importance_model.pt"
    else:
        opt_frame_importance_model_fp = "./models/opt_frame_importance_model_no_audio.pt"
        ckp_frame_importance_model_fp = "./models/ckp_frame_importance_model_no_audio.pt"
    annotation_fp = 'ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv'
    audio_fp = './tmp/audio.wav'
    h5_file_path = 'ydata-tvsum50-v1_1/ground_truth/eccv16_dataset_tvsum_google_pool5.h5'
    mat_file_path = 'ydata-tvsum50-v1_1/ground_truth/ydata-tvsum50.mat'
    video_fps = sorted(glob.glob('./ydata-tvsum50-v1_1/video/*.mp4'), reverse = False)

    # Hyperparameters (preprocessing)
    videos_limitation = -1
    skip_frames = 15

    # Hyperparameters (training process - frame importance model)
    num_epochs = 20 # 100
    lr = 0.00008

    ## ! Initialization: End

    video_fps = video_fps[:videos_limitation]

    ground_truths_trimmed = []
    ground_truths_full = []
    frames = []
    audios = []
    full_n_frames = []
    for video_fp in video_fps:
        video_id = video_fp.split('/')[-1].split('.')[0]
        print("Extracting content from %s"%(video_id))
        ground_truth_trimmed, ground_truth_full = get_annotations(annotation_fp = annotation_fp, video_id = video_id, skip_frames = skip_frames)
        ground_truths_trimmed.append(ground_truth_trimmed)
        ground_truths_full.append(ground_truth_full)
        visual_frames_tensor, full_n_frames_ = extract_condensed_frame_tensor(video_fp, skip_frames = skip_frames)
        N = len(visual_frames_tensor)
        export_audio_from_video(audio_fp = audio_fp, video_fp = video_fp)
        audio_features_tensor = extract_audio_features(audio_fp = audio_fp, n_frames = N)
        full_n_frames.append(full_n_frames_)
        frames.append(visual_frames_tensor)
        audios.append(audio_features_tensor)

    val_dataset = dataloader(fps = [video_fps.pop()], frames = [frames.pop()], full_n_frames = [full_n_frames.pop()], audios = [audios.pop()], labels = [ground_truths_trimmed.pop()])
    train_dataset = dataloader(fps = video_fps, frames = frames, full_n_frames = full_n_frames, audios = audios, labels = ground_truths_trimmed)
    del frames, audios, ground_truths_trimmed

    print("Number of train videos: %d"%(len(train_dataset)))
    print("Number of val videos: %d"%(len(val_dataset)))

    frame_importance_model = audio_visual_model(audio_included = audio_included)
    if load_ckp:
        frame_importance_model.load_state_dict(torch.load(f = ckp_frame_importance_model_fp))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(params = frame_importance_model.parameters(), lr = lr)

    est_train_losses = []
    est_train_f_scores_avg = []
    est_train_f_scores_max = []
    val_losses = []
    val_f_scores_avg = []
    val_f_scores_max = []

    # ! Evaluation of model prior to training: Begin

    print("\n[Initial model evaluation]\n")
    t0 = time()

    train_losses = []
    train_f_scores_avg = []
    train_f_scores_max = []
    for video_idx, (video_id, batch_frames, batch_audios, batch_labels) in enumerate(train_dataset):
        with torch.no_grad():
            batch_predictions = frame_importance_model(batch_audios, batch_frames)
            train_losses.append(criterion(batch_predictions, batch_labels).item())
            full_n_batch_frames = train_dataset.full_n_frames_
            batch_f_score_avg, batch_f_score_max = get_train_f_scores()
            print("Video: %d/%d\nBatch Set - Size: %d - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f"%(video_idx, len(train_dataset)-1, full_n_batch_frames, train_losses[-1], batch_f_score_avg, batch_f_score_max))
            train_f_scores_avg.append(batch_f_score_avg)
            train_f_scores_max.append(batch_f_score_max)
        torch.cuda.empty_cache()

    video_id, val_frames, val_audios, val_labels = next(iter(val_dataset))

    print("Validation video: %s"%(video_id))

    with torch.no_grad():
        val_predictions = frame_importance_model(val_audios, val_frames)
        val_loss = criterion(val_predictions, val_labels).item()
        full_n_val_frames = val_dataset.full_n_frames_
        val_f_score_avg, val_f_score_max = get_val_f_scores()

    train_est_loss = sum(train_losses) / len(train_losses)
    est_train_f_score_avg = sum(train_f_scores_avg) / len(train_f_scores_avg)
    est_train_f_score_max = sum(train_f_scores_max) / len(train_f_scores_max)

    opt_val_loss = val_loss
    opt_est_train_loss = train_est_loss
    opt_epoch = -1
    opt_est_train_f_score_avg = est_train_f_score_avg
    opt_est_train_f_score_max = est_train_f_score_max
    opt_val_f_score_avg = val_f_score_avg
    opt_val_f_score_max = val_f_score_max

    est_train_losses.append(train_est_loss)
    est_train_f_scores_avg.append(est_train_f_score_avg)
    est_train_f_scores_max.append(est_train_f_score_max)
    val_losses.append(val_loss)
    val_f_scores_avg.append(val_f_score_avg)
    val_f_scores_max.append(val_f_score_max)

    t1 = time()
    print("Train Set - Est. loss: %.4f - Est. F-score Avg: %.4f - Est. F-score Max: %.4f\nVal Set - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f\nΔt: %.1fs"%(train_est_loss, est_train_f_score_avg, est_train_f_score_max, val_loss, val_f_score_avg, val_f_score_max, t1-t0))

    # ! Evaluation of model prior to training: End

    # Training
    print("\n[Training states]\n")
    opt_epoch = -1
    t0_train = time()
    for epoch in range(num_epochs):
        batch_losses = []
        train_f_scores_avg = []
        train_f_scores_max = []
        t0_epoch = time()
        print("Epoch: %d/%d"%(epoch, num_epochs-1))
        for video_idx, (video_id, batch_frames, batch_audios, batch_labels) in enumerate(train_dataset):

            t0_step = time()

            # Train step
            optimizer.zero_grad()
            batch_predictions = frame_importance_model(batch_audios, batch_frames)
            batch_loss = criterion(batch_predictions, batch_labels)
            batch_loss.backward()
            optimizer.step()
            full_n_batch_frames = train_dataset.full_n_frames_
            batch_losses.append(batch_loss.item())

            batch_f_score_avg, batch_f_score_max = get_train_f_scores()

            train_f_scores_avg.append(batch_f_score_avg)
            train_f_scores_max.append(batch_f_score_max)

            # Val scores
            video_id, val_frames, val_audios, val_labels = next(iter(val_dataset))
            with torch.no_grad():
                val_predictions = frame_importance_model(val_audios, val_frames)
                val_loss = criterion(val_predictions, val_labels).item()
                full_n_val_frames = val_dataset.full_n_frames_
                val_f_score_avg, val_f_score_max = get_val_f_scores()

            torch.cuda.empty_cache()

            t1_step = time()

            print("Video: %d/%d\nBatch Set - Size: %d - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f\nVal Set - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f\nΔt: %.1fs"%(video_idx, len(train_dataset)-1, full_n_batch_frames, batch_losses[-1], batch_f_score_avg, batch_f_score_max, val_loss, val_f_score_avg, val_f_score_max, t1_step-t0_step))
        train_est_loss = sum(batch_losses) / len(batch_losses)
        est_train_f_score_avg = sum(train_f_scores_avg) / len(train_f_scores_avg)
        est_train_f_score_max = sum(train_f_scores_max) / len(train_f_scores_max)

        est_train_losses.append(train_est_loss)
        est_train_f_scores_avg.append(est_train_f_score_avg)
        est_train_f_scores_max.append(est_train_f_score_max)
        val_losses.append(val_loss)
        val_f_scores_avg.append(val_f_score_avg)
        val_f_scores_max.append(val_f_score_max)

        print("\n" + color.BOLD + "Overall epoch state" + color.END + "\n")

        if val_loss < opt_val_loss:
            print("Val ΔL " + color.GREEN + "↓ %.4f"%(abs(val_loss - opt_val_loss)) + color.END)
            opt_val_loss = val_loss
            opt_est_train_loss = train_est_loss
            opt_epoch = epoch
            opt_est_train_f_score_avg = est_train_f_score_avg
            opt_est_train_f_score_max = est_train_f_score_max
            opt_val_f_score_avg = val_f_score_avg
            opt_val_f_score_max = val_f_score_max
            torch.save(obj = frame_importance_model.state_dict(), f = opt_frame_importance_model_fp)
        else:
            print("Val ΔL " + color.RED + "↑ %.4f"%(abs(val_loss - opt_val_loss)) + color.END)

        torch.save(obj = frame_importance_model.state_dict(), f = ckp_frame_importance_model_fp)
        generate_metric_plots(opt_val_loss, est_train_losses, est_train_f_scores_avg, est_train_f_scores_max, val_losses, val_f_scores_avg, val_f_scores_max)

        t1_epoch = time()
        print("Train Set - Est. loss: %.4f - Est. F-score Avg: %.4f - Est. F-score Max: %.4f\nVal Set - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f\nΔt: %.1fs"%(train_est_loss, est_train_f_score_avg, est_train_f_score_max, val_loss, val_f_score_avg, val_f_score_max, t1_epoch - t0_epoch))
        print()

    t1_train = time()

    print("[Final model evaluation]\n")
    print("Optimal epoch: %d"%(opt_epoch))
    print("Train Set - Est. loss: %.4f - Est. F-score Avg: %.4f - Est. F-score Max: %.4f\nVal Set - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f\nΔt: %.1fs"%(opt_est_train_loss, opt_est_train_f_score_avg, opt_est_train_f_score_max, opt_val_loss, opt_val_f_score_avg, opt_val_f_score_max, t1_train - t0_train))
    print("\nOperation completed")

def infer(video_fp: str, audio_included: bool):

    # Paths
    if audio_included:
        opt_frame_importance_model_fp = "./models/opt_frame_importance_model.pt"
    else:
        opt_frame_importance_model_fp = "./models/opt_frame_importance_model_no_audio.pt"
    audio_fp = './tmp/audio.wav'
    h5_file_path = 'ydata-tvsum50-v1_1/ground_truth/eccv16_dataset_tvsum_google_pool5.h5'
    mat_file_path = 'ydata-tvsum50-v1_1/ground_truth/ydata-tvsum50.mat'

    skip_frames = 15

    print("Input video:\n", video_fp)

    video_frames, full_n_val_frames = extract_condensed_frame_tensor(video_fp, skip_frames = skip_frames)
    visual_frames_tensor = torch.tensor(video_frames, dtype = torch.float32)
    full_val_frames = get_frame_tensor(video_fp)

    export_audio_from_video(audio_fp = audio_fp, video_fp = video_fp)
    N = len(visual_frames_tensor)
    audio_features_tensor = torch.tensor(extract_audio_features(audio_fp = audio_fp, n_frames = N), dtype = torch.float32)

    data = dataloader(fps = [video_fp], frames = [visual_frames_tensor], full_n_frames = [full_n_val_frames], audios = [audio_features_tensor], labels = None)

    frame_importance_model = audio_visual_model(audio_included = audio_included)
    frame_importance_model.load_state_dict(torch.load(f = opt_frame_importance_model_fp))

    video_id, val_frames, val_audios, _ = next(iter(data))
    full_val_frames = get_frame_tensor(fp = video_fp)

    val_predictions = frame_importance_model(val_audios, val_frames)

    full_n_val_frames = data.full_n_frames_

    # Clips and Knapsack optimization
    summarized_video, summarized_video_frame_indices = postprocessing\
    (
        video_id = video_id,
        h5_file_path = h5_file_path,
        mat_file_path = mat_file_path,
        batch_predictions = val_predictions,
        skip_frames = skip_frames,
        full_n_frames = full_n_val_frames,
        full_frames = full_val_frames
    )

    # Summarization evaluation
    # f_score_avg, f_score_max = evaluation_method(ground_truth_path = mat_file_path, summary_indices = summarized_video_frame_indices, video_id = video_fp.split('/')[-1].split('.')[0])

    # print('F-score Avg: %.4f'%(f_score_avg))
    # print('F-score Max: %.4f'%(f_score_max))

    export_video(frames = summarized_video, output_path = "./tmp/%s.mp4"%(data.title), frame_rate = 30)
    print("\n[Exported video details]\n\nID: %s\nTitle: %s"%(video_id, data.title))


if __name__ == '__main__':

    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    if not os.path.exists('models'):
        os.mkdir('models')

    load_ckp = False
    if len(sys.argv) == 3:
        assert "--train" in sys.argv[1] and sys.argv[2] == "--checkpoint", "E: Invalid prompt arguments"
        load_ckp = True

    infer_video_fp = 'ydata-tvsum50-v1_1/video/-esJrBWj2d8.mp4'
    if len(sys.argv) == 2 and sys.argv[1] == '--train':
        train_importance_model(audio_included = True, load_ckp = load_ckp)
    elif len(sys.argv) == 2 and sys.argv[1] == '--train-no-audio':
        train_importance_model(audio_included = False, load_ckp = load_ckp)
    elif len(sys.argv) == 2 and sys.argv[1] == '--infer':
        infer(video_fp = infer_video_fp, audio_included = True)
    elif len(sys.argv) == 2 and sys.argv[1] == '--infer-no-audio':
        infer(video_fp = infer_video_fp, audio_included = False)