import torch
import os
import sys
import torch
import glob
import torch.optim as optim
from time import time
import torch.nn as nn
import numpy as np

from utils import extract_condensed_frame_tensor, export_audio_from_video, extract_audio_features, dataloader, AVM, export_video, get_frame_tensor, postprocess, get_dataloaders, postprocess_and_get_fscores
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

    ## ! Initialization: Begin

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
    # video_fps = sorted(glob.glob('./ydata-tvsum50-v1_1/video/*.mp4'), reverse = False)
    video_fps = ['./ydata-tvsum50-v1_1/video/37rzWOQsNIw.mp4', './ydata-tvsum50-v1_1/video/RBCABdttQmI.mp4']

    # Hyperparameters (preprocessing)
    skip_frames = 30

    # Hyperparameters (training process - frame importance model)
    num_epochs = 25 # 100
    lr = 0.00001
    train_ratio = 0.8
    np.random.seed(seed = 12344321)

    ## ! Initialization: End

    # np.random.shuffle(video_fps)

    train_dataset, val_dataset = get_dataloaders(video_fps = video_fps, skip_frames = skip_frames, train_ratio = train_ratio, annotation_fp = annotation_fp, mat_file_path = mat_file_path, h5_file_path = h5_file_path)

    print("Number of train videos: %d"%(len(train_dataset)))
    print("Number of val videos: %d"%(len(val_dataset)))

    frame_importance_model = AVM(audio_included = audio_included)
    if load_ckp:
        frame_importance_model.load_state_dict(torch.load(f = ckp_frame_importance_model_fp))

    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
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

    train_losses_step = []
    train_f_scores_avg_step = []
    train_f_scores_max_step = []

    val_losses_step = []
    val_f_scores_avg_step = []
    val_f_scores_max_step = []

    for video_idx, (video_id, batch_frames, batch_audios, batch_labels, batch_gd_summarized_video_frame_indices_) in enumerate(train_dataset):
        with torch.no_grad():
            batch_predictions = frame_importance_model(batch_audios, batch_frames)
            # train_loss = criterion(batch_predictions, (batch_labels-1).long()).item()
            # batch_predictions = torch.argmax(batch_predictions, axis = 1) + 1
            train_loss = criterion(batch_predictions, batch_labels).item()
            full_n_batch_frames = train_dataset.full_n_frames_
            batch_f_score_avg, batch_f_score_max = postprocess_and_get_fscores(video_id = video_id, batch_predictions = batch_predictions, full_n_batch_frames = full_n_batch_frames, gd_summarized_video_frame_indices = batch_gd_summarized_video_frame_indices_, h5_file_path = h5_file_path, mat_file_path = mat_file_path, skip_frames = skip_frames)
            # print("Video: %d/%d - ID: %s\nBatch Set - Size: %d - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f"%(video_idx, len(train_dataset)-1, video_id, full_n_batch_frames, train_loss, batch_f_score_avg, batch_f_score_max))
            train_losses_step.append(train_loss)
            train_f_scores_avg_step.append(batch_f_score_avg)
            train_f_scores_max_step.append(batch_f_score_max)
        torch.cuda.empty_cache()

    with torch.no_grad():
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

    # del train_losses_step, train_f_scores_avg_step, train_f_scores_max_step, val_losses_step, val_f_scores_avg_step, val_f_scores_max_step

    opt_epoch = -1
    opt_est_train_loss = est_train_loss
    opt_est_train_f_score_avg = est_train_f_score_avg
    opt_est_train_f_score_max = est_train_f_score_max
    opt_val_loss = val_loss
    opt_val_f_score_avg = val_f_score_avg
    opt_val_f_score_max = val_f_score_max

    est_train_losses.append(est_train_loss)
    est_train_f_scores_avg.append(est_train_f_score_avg)
    est_train_f_scores_max.append(est_train_f_score_max)
    val_losses.append(val_loss)
    val_f_scores_avg.append(val_f_score_avg)
    val_f_scores_max.append(val_f_score_max)

    t1 = time()
    print("Train Set - Est. loss: %.4f - Est. F-score Avg: %.4f - Est. F-score Max: %.4f\nVal Set - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f\nΔt: %.1fs"%(est_train_loss, est_train_f_score_avg, est_train_f_score_max, val_loss, val_f_score_avg, val_f_score_max, t1-t0))

    # ! Evaluation of model prior to training: End

    # Training
    print("\n[Training states]\n")
    opt_epoch = -1
    t0_train = time()
    for epoch in range(num_epochs):

        prev_val_loss = val_loss

        train_losses_step = []
        train_f_scores_avg_step = []
        train_f_scores_max_step = []

        val_losses_step = []
        val_f_scores_avg_step = []
        val_f_scores_max_step = []

        t0_epoch = time()

        print(color.BOLD + "Epoch %d/%d"%(epoch, num_epochs-1) + color.END + "\n")
        for video_idx, (video_id, batch_frames, batch_audios, batch_labels, batch_gd_summarized_video_frame_indices_) in enumerate(train_dataset):

            t0_step = time()

            # Train step
            optimizer.zero_grad()
            batch_predictions = frame_importance_model(batch_audios, batch_frames)
            # batch_loss = criterion(batch_predictions, (batch_labels-1).long())
            # batch_predictions = torch.argmax(batch_predictions, axis = 1) + 1
            batch_loss = criterion(batch_predictions, batch_labels)
            batch_loss.backward()
            optimizer.step()
            full_n_batch_frames = train_dataset.full_n_frames_

            # print(torch.sum(frame_importance_model.visbl.conv1.weight.grad).item())

            batch_f_score_avg, batch_f_score_max = postprocess_and_get_fscores(video_id = video_id, batch_predictions = batch_predictions, full_n_batch_frames = full_n_batch_frames, gd_summarized_video_frame_indices = batch_gd_summarized_video_frame_indices_, h5_file_path = h5_file_path, mat_file_path = mat_file_path, skip_frames = skip_frames)

            train_losses_step.append(batch_loss.item())
            train_f_scores_avg_step.append(batch_f_score_avg)
            train_f_scores_max_step.append(batch_f_score_max)

            t1_step = time()
            # print("Video: %d/%d - ID: %s\nBatch Set - Size: %d - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f\nVal Set - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f\nΔt: %.1fs"%(video_idx, len(train_dataset)-1, video_id, full_n_batch_frames, batch_loss.item(), batch_f_score_avg, batch_f_score_max, val_loss, val_f_score_avg, val_f_score_max, t1_step-t0_step))

            torch.cuda.empty_cache()

        # Val scores
        with torch.no_grad():

            for val_video_id, val_frames, val_audios, val_labels, val_gd_summarized_video_frame_indices_ in val_dataset:

                val_predictions = frame_importance_model(val_audios, val_frames)
                # val_loss = criterion(val_predictions, (val_labels-1).long()).item()
                # val_predictions = torch.argmax(val_predictions, axis = 1) + 1
                val_loss = criterion(val_predictions, val_labels).item()
                full_n_val_frames = val_dataset.full_n_frames_
                val_f_score_avg, val_f_score_max = postprocess_and_get_fscores(video_id = val_video_id, batch_predictions = val_predictions, full_n_batch_frames = full_n_val_frames, gd_summarized_video_frame_indices = val_gd_summarized_video_frame_indices_, h5_file_path = h5_file_path, mat_file_path = mat_file_path, skip_frames = skip_frames)

                val_losses_step.append(val_loss)
                val_f_scores_avg_step.append(val_f_score_avg)
                val_f_scores_max_step.append(val_f_score_max)

        est_train_loss = sum(train_losses_step) / len(train_losses_step)
        est_train_f_score_avg = sum(train_f_scores_avg_step) / len(train_f_scores_avg_step)
        est_train_f_score_max = sum(train_f_scores_max_step) / len(train_f_scores_max_step)

        est_train_losses.append(est_train_loss)
        est_train_f_scores_avg.append(est_train_f_score_avg)
        est_train_f_scores_max.append(est_train_f_score_max)

        val_loss = sum(val_losses_step) / len(val_losses_step)
        val_f_score_avg = sum(val_f_scores_avg_step) / len(val_f_scores_avg_step)
        val_f_score_max = sum(val_f_scores_max_step) / len(val_f_scores_max_step)

        val_losses.append(val_loss)
        val_f_scores_avg.append(val_f_score_avg)
        val_f_scores_max.append(val_f_score_max)

        print("\n" + color.BOLD + "Overall epoch state:" + color.END)

        if val_loss < prev_val_loss:
            print("Val ΔL " + color.GREEN + "↓ %.4f"%(abs(val_loss - prev_val_loss)) + color.END)
        else:
            print("Val ΔL " + color.RED + "↑ %.4f"%(abs(val_loss - prev_val_loss)) + color.END)
        if val_loss < opt_val_loss:
            opt_val_loss = val_loss
            opt_est_train_loss = est_train_loss
            opt_epoch = epoch
            opt_est_train_f_score_avg = est_train_f_score_avg
            opt_est_train_f_score_max = est_train_f_score_max
            opt_val_f_score_avg = val_f_score_avg
            opt_val_f_score_max = val_f_score_max
            torch.save(obj = frame_importance_model.state_dict(), f = opt_frame_importance_model_fp)

        torch.save(obj = frame_importance_model.state_dict(), f = ckp_frame_importance_model_fp)

        assert len(est_train_losses) == len(est_train_f_scores_avg) == len(est_train_f_scores_max) == len(val_losses) == len(val_f_scores_avg) == len(val_f_scores_max), 'E: Inconsistent score list lengths'

        generate_metric_plots(opt_val_loss, est_train_losses, est_train_f_scores_avg, est_train_f_scores_max, val_losses, val_f_scores_avg, val_f_scores_max, exported_image_fp = exported_image_fp)

        t1_epoch = time()

        print("Train Set - Est. loss: %.4f - Est. F-score Avg: %.4f - Est. F-score Max: %.4f\nVal Set - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f\nΔt: %.1fs"%(est_train_loss, est_train_f_score_avg, est_train_f_score_max, val_loss, val_f_score_avg, val_f_score_max, t1_epoch - t0_epoch))
        print()

    t1_train = time()

    print("[Final model evaluation]\n")
    print("Optimal epoch: %d"%(opt_epoch))
    print("Train Set - Est. loss: %.4f - Est. F-score Avg: %.4f - Est. F-score Max: %.4f\nVal Set - Loss: %.4f - F-score Avg: %.4f - F-score Max: %.4f - Improvement: %.4f\nΔt: %.1fs"%(opt_est_train_loss, opt_est_train_f_score_avg, opt_est_train_f_score_max, opt_val_loss, opt_val_f_score_avg, opt_val_f_score_max, opt_est_train_f_score_avg - est_train_f_scores_avg[0], t1_train - t0_train))
    print("\nOperation completed")

def infer(video_fp: str, audio_included: bool):

    # Paths
    if audio_included:
        opt_frame_importance_model_fp = "./models/opt_frame_importance_model.pt"
    else:
        opt_frame_importance_model_fp = "./models/opt_frame_importance_model_no_audio.pt"
    h5_file_path = 'ydata-tvsum50-v1_1/ground_truth/eccv16_dataset_tvsum_google_pool5.h5'
    mat_file_path = 'ydata-tvsum50-v1_1/ground_truth/ydata-tvsum50.mat'
    audio_fp = ".".join(video_fp.split(".")[:-1]) + ".wav"

    skip_frames = 60 # 2-seconds per annotation

    print("Input video:\n", video_fp)

    video_frames, full_n_val_frames = extract_condensed_frame_tensor(video_fp, skip_frames = skip_frames)
    visual_frames_tensor = torch.tensor(video_frames, dtype = torch.float32)
    full_val_frames = get_frame_tensor(video_fp)

    export_audio_from_video(audio_fp = audio_fp, video_fp = video_fp)
    N = len(visual_frames_tensor)
    audio_features_tensor = torch.tensor(extract_audio_features(audio_fp = audio_fp, n_frames = N), dtype = torch.float32)

    data = dataloader(fps = [video_fp], frames = [visual_frames_tensor], full_n_frames = [full_n_val_frames], audios = [audio_features_tensor], labels = None, gd_summarized_video_frame_indices = None)

    frame_importance_model = AVM(audio_included = audio_included)
    frame_importance_model.load_state_dict(torch.load(f = opt_frame_importance_model_fp))

    video_id, val_frames, val_audios, _, _ = next(iter(data))
    full_val_frames = get_frame_tensor(fp = video_fp)

    val_predictions = frame_importance_model(val_audios, val_frames)

    full_n_val_frames = data.full_n_frames_

    # Clips and Knapsack optimization
    summarized_video, summarized_video_frame_indices = postprocess\
    (
        video_id = video_id,
        h5_file_path = h5_file_path,
        mat_file_path = mat_file_path,
        batch_importances = val_predictions,
        skip_frames = skip_frames,
        full_n_frames = full_n_val_frames,
        full_frames = full_val_frames
    )

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
    else:
        exit("E: Invalid user prompt")