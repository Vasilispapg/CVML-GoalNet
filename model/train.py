import sys
sys.path.append('model')

from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from model import audio_visual_model
from dataloader import dataloader


def train(frames, audios, ground_truths):

    num_epochs = 100
    val_dataset = dataloader(frames = [frames.pop()], audios = [audios.pop()], labels = [ground_truths.pop()])
    train_dataset = dataloader(frames = frames, audios = audios, labels = ground_truths)
    del frames, audios, ground_truths

    print("Number of train videos: %d"%(len(train_dataset)))
    print("Number of val videos: %d"%(len(val_dataset)))

    avm = audio_visual_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params = avm.parameters(), lr = 0.00008)

    val_frames, val_audios, val_labels = val_dataset[0]

    # Evaluation of model prior to training
    print("\n[Initial model evaluation]")
    t0 = time()
    initial_train_losses = []
    for video_idx, (batch_frames, batch_audios, batch_labels) in enumerate(train_dataset):
        with torch.no_grad():
            initial_train_losses.append(criterion(avm(batch_audios, batch_frames), batch_labels).item())
        torch.cuda.empty_cache()
    initial_train_loss = sum(initial_train_losses) / len(initial_train_losses)
    with torch.no_grad():
        initial_val_loss = criterion(avm(val_audios, val_frames), batch_labels).item()
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
            batch_predictions = avm(batch_audios, batch_frames)
            batch_loss = criterion(batch_predictions, batch_labels)
            batch_loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_loss = criterion(avm(val_audios, val_frames), val_labels).item()
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
    return batch_predictions