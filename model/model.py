import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
# dn eimai sigoyros
from xception import xception
from torch.utils.data import Dataset,DataLoader
from time import time
from torch.utils.data.dataset import random_split
from torchsummary import summary
import os


def display_tensor_info(tnsr, var_name):
    print('---')
    print('variable name: %s'%(var_name))
    print('Shape: ')
    print(tuple(tnsr.shape))
    print()
    print('Mean: ')
    print(torch.mean(tnsr).item())
    print()
    print('Variance: ')
    print(torch.var(tnsr).item())
    print()
    print('Min, Max: ')
    print(torch.min(tnsr).item(), torch.max(tnsr).item())
    print()
    print('Exists NAN')
    print(True in torch.isnan(tnsr))
    if True in torch.isnan(tnsr):
        exit('ERROR')
    print('---')

# The `AudioVisualModel` class defines a neural network model that combines audio and visual features
# extracted from MFCC and Xception models respectively, and makes a decision using softmax activation.
class AudioVisualModel(nn.Module):
    def __init__(self):
        super(AudioVisualModel, self).__init__()
        # Visual Branch (Xception)
        self.visual_model = xception(num_classes=1000)
        self.visual_model.fc = nn.Identity()  # Adapt final layer based on Xception architecture
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        

        # Audio Branch (Simple CNN for MFCC)
        self.audio_model = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),  # Adjust size calculation based on your architecture
            nn.ReLU()
        )
        
        # OUTPUT SHAPE (22,5)

        # Fusion and Decision Making using softmax
        self.fusion = nn.LazyLinear(5)

    def forward(self, audio_input, visual_input): # audio_input: MFCC features, visual_input: Frames
        # display_tensor_info(tnsr = visual_input, var_name = 'visual_input')
        # AUDIO FEATURES
        audio_features = self.audio_model(audio_input)
        # VISUAL FEATURES
        with torch.no_grad():
            xception_features = self.visual_model(visual_input)  # Extract features using Xception
        
        visual_features = nn.ReLU(inplace=True)(xception_features)
        visual_features = F.adaptive_avg_pool2d(visual_features, (1, 1))
        visual_features = visual_features.view(visual_features.size(0), -1)
        visual_features = nn.LazyLinear(512).to(visual_features.device)(visual_features)
        # visual_features = self.visual_model_expand(xception_features)  # Further process features with shape: [batch_size, feature_dim_visual]

        # FUSE
        combined_features = torch.cat((audio_features, visual_features), axis = -1)

        output = self.fusion(combined_features)
        output = F.softmax(output, dim=1)
        return output

def saveModel(model, path):
    torch.save(model.state_dict(), path)
    
def loadModel(model, path):
    if(not os.path.exists(path)):
        return None
    model.load_state_dict(torch.load(path))
    return model


def callNN(sample_visual_frames, audio_features, labels,test_dataset,val_dataset):  
    """
    This function trains an audio-visual model using provided data and then uses the model to produce
    labels for full video frames.
    
    :param sample_visual_frames: Sample visual frames for training the model
    :param audio_features: Audio features extracted from the audio of the video frames
    :param labels: The `labels` parameter in the `callNN` function represents the ground truth labels
    for the training data. These labels are used during the training process to calculate the loss and
    optimize the model parameters. The function seems to be a part of a neural network training pipeline
    where the model is trained on visual
    :param test_dataset: The `test_dataset` parameter in the `callNN` function seems to represent a
    tuple containing three elements:
    :param val_dataset: The `val_dataset` parameter in the `callNN` function refers to the validation
    dataset that is used to evaluate the performance of the model during training. It typically consists
    of visual frames, audio features, and corresponding labels for validation purposes. This dataset
    helps in monitoring the model's performance on unseen data
    :return: The function `callNN` is returning the `labels` that are produced by the model after
    training and testing on the provided datasets.
    """
    
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # dataset for train and full_video to produce labels
    dataset = DataLoaderFrameLabeled(frames = sample_visual_frames, audio = audio_features, labels = labels)
    full_video_dataloader=DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Generate test and validation datasets
    dataset_test = DataLoaderFrameLabeled(frames = test_dataset[0], audio = test_dataset[1], labels = test_dataset[2])
    dataset_val = DataLoaderFrameLabeled(frames = val_dataset[0], audio = val_dataset[1], labels = val_dataset[2])
    
    # Init the model
    avm = loadModel(AudioVisualModel(), os.path.join(os.getcwd(), 'model.pth'))
    if(avm == None):
        print("Creating a new model")
        avm = AudioVisualModel()
    avm = avm.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(avm.parameters(), lr=0.0002)
    
    train_dataset, val_dataset, test_dataset = GenerateDatasets(dataset,dataset_test,dataset_val)
    
    # Train the model
    avm = trainMode(avm, train_dataset, val_dataset, num_epochs, optimizer, criterion, device)
    
    # Test the model
    print('Testing')
    avm = testMode(avm, test_dataset, criterion, device)        

    saveModel(avm, os.path.join(os.getcwd(), 'model.pth'))
    print(f"File saved at {os.getcwd()}/model.pth")
    
    print("Producing Labels")
    labels=produceLabels(avm, full_video_dataloader, device)
    
    return labels
    
def produceLabels(avm, dataloader,device):
    avm.eval()
    with torch.no_grad():
        labels = []
        for (frames, audio, _) in dataloader:
            frames = frames.to(device)
            audio = audio.to(device)
            output = avm(audio, frames)
            _, predicted_indices = torch.max(output.data, 1)
            labels.extend(predicted_indices)
                
    return  [l.to('cpu') for l in labels]

def trainMode(avm, train_dataset, val_dataset, num_epochs, optimizer, criterion, device="cpu"):

    avm.train()
    time_ = time()
    print("Training Started")
    for epoch in range(num_epochs):
        for i, (frames, audio, labels) in enumerate(train_dataset):
            frames = frames.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = avm(audio, frames)
            # output = torch.argmax(output, axis=1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {loss.item()} Time: {time()-time_}")
            torch.cuda.empty_cache()

        # validation
        print('Training')
        testMode(avm, train_dataset, criterion, device)
        print("Validation")
        avm = testMode(avm, val_dataset, criterion, device)
    return avm

def testMode(avm, dataloader, criterion, device="cpu"):
    avm.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (frames, audio, labels) in dataloader:
            frames = frames.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            
            output = avm(audio, frames)
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
        print('-'*30)
        print(f"Loss: {loss.item()}, Accuracy: {acc}")
        print('-'*30)
        
            
    return avm
    
def GenerateDatasets(dataset,dataset_test,dataset_val):
    """
    The function `GenerateDatasets` creates data loaders for training, validation, and testing datasets
    with specified batch sizes and shuffle settings.
    
    :param dataset: The `dataset` parameter is typically a dataset object that contains the training
    data samples. This dataset object is used to create a DataLoader for the training set
    :param dataset_test: It seems like you were about to provide some information about the
    `dataset_test` parameter but the message got cut off.
    :param dataset_val: It seems like you were about to provide some information about the `dataset_val`
    parameter, but the message got cut off. Could you please provide more details or let me know how I
    can assist you further with the `dataset_val` parameter?
    :return: The function `GenerateDatasets` returns a list containing three DataLoader objects:
    `train_loader`, `val_loader`, and `test_loader`. These DataLoader objects are used to load batches
    of data for training, validation, and testing purposes.
    """

    batch_size = 64
    # Ensure the test set gets any remaining samples after integer division

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    return [train_loader, val_loader, test_loader]

# This class `DataLoaderFrameLabeled` is a custom dataset class in PyTorch for loading labeled visual
# frames and audio features.
class DataLoaderFrameLabeled(Dataset):
    def __init__(self, frames, audio, labels):
        self.labels = torch.LongTensor(labels)
        self.visual_frames_tensor = torch.stack([torch.tensor(frame, dtype=torch.float).permute(2, 0, 1) for frame in frames]  )
        self.audio_features_tensor= torch.tensor(audio,dtype=torch.float)

    def __len__(self):
        return len(self.labels)
    
    def normalize_visual_input(self, tensor):
        # Normalize based on the Xception model's expected values
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float).view(3, 1, 1)
        return (tensor - mean) / (std + 0.001)

    def __getitem__(self, idx):
        visual_frame_tensor = self.visual_frames_tensor[idx]
        visual_frame_tensor = self.normalize_visual_input(visual_frame_tensor)
        audio_feature_tensor =  self.audio_features_tensor[idx]
        label = self.labels[idx]

        return visual_frame_tensor, audio_feature_tensor, label
 
