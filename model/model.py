import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
# dn eimai sigoyros
from xception import xception
from torch.utils.data import Dataset,DataLoader
from time import time
from torch.utils.data.dataset import random_split


# The `AudioVisualModel` class defines a neural network model that combines audio and visual features
# extracted from MFCC and Xception models respectively, and makes a decision using softmax activation.
class AudioVisualModel(nn.Module):
    def __init__(self):
        super(AudioVisualModel, self).__init__()
        # Visual Branch (Xception)
        self.visual_model = xception(num_classes=1000)
        self.visual_model.fc = nn.Identity()  # Adapt final layer based on Xception architecture
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # as input shape (2048, 10, 10) from Xception
        self.visual_model_expand = nn.Sequential(
            # INPUT SHAPE (2048, 10, 10)
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
        )
        # OUTPUT SHAPE (128)

        # Audio Branch (Simple CNN for MFCC)
        self.audio_model = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(5),  # Adjust size calculation based on your architecture
            nn.ReLU()
        )
        # OUTPUT SHAPE (22,5)

        # Fusion and Decision Making using softmax
        self.fusion = nn.LazyLinear(5)

    def forward(self, audio_input, visual_input): # audio_input: MFCC features, visual_input: Frames
        # AUDIO FEATURES
        audio_features = self.audio_model(audio_input)
        # VISUAL FEATURES
        xception_features = self.visual_model(visual_input)  # Extract features using Xception
        visual_features = self.visual_model_expand(xception_features)  # Further process features with shape: [batch_size, feature_dim_visual]
        combined_features = torch.cat((audio_features, visual_features), axis = -1)
        output = self.fusion(combined_features)
        output = F.softmax(output, dim=1)
        return output
    

def callNN(sample_visual_frames, audio_features, labels):
    # labels is a list with size n_annotators, where each list contains the importance with size n_all_frames
    # Hyperparameters    
    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    # Initialize your model
    # Refine descriptors - avm: Audio visual model
    
    dataset = DataLoaderFrameLabeled(frames = sample_visual_frames, audio = audio_features, labels = labels)
    
    avm = AudioVisualModel().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(avm.parameters())
    
    train_dataset, val_dataset, test_dataset = splitDataset(dataset)
    
    # Train the model
    avm = trainMode(avm, train_dataset, val_dataset, num_epochs, optimizer, criterion, device)
    
    # Test the model
    avm = testMode(avm, test_dataset, criterion, device)        
        
    
    # Convert frames and audio features to PyTorch tensors

    # print(predictions)
    

def trainMode(avm, train_dataset, val_dataset, num_epochs, optimizer, criterion, device):
    """
    The function `trainMode` trains a model using a specified optimizer and criterion over a specified
    number of epochs on a training dataset, with periodic validation using a separate validation
    dataset.
    
    :param avm: The `avm` parameter in the `trainMode` function is typically a neural network model that
    you want to train. It stands for "Audio-Visual Model" in this context. This model is trained using
    the provided `train_dataset` and validated using the `val_dataset` for a specified
    :param train_dataset: The `train_dataset` parameter typically refers to the dataset used for
    training the model. It contains the training samples along with their corresponding labels. This
    dataset is used to update the model's parameters during the training process through backpropagation
    :param val_dataset: The `val_dataset` parameter in the `trainMode` function is typically a dataset
    used for validation during the training process. It is a separate dataset from the training dataset
    and is used to evaluate the model's performance on unseen data and prevent overfitting. The
    validation dataset is not used for training
    :param num_epochs: The `num_epochs` parameter specifies the number of times the model will iterate
    over the entire training dataset during the training process. It essentially determines how many
    times the model will learn from the training data to improve its performance before the training
    process is completed
    :param optimizer: The optimizer parameter in the trainMode function is typically an instance of an
    optimization algorithm such as SGD (Stochastic Gradient Descent), Adam, or RMSprop. It is used to
    update the parameters of the model during training in order to minimize the loss function
    :param criterion: The `criterion` parameter in the `trainMode` function is typically used to define
    the loss function that will be optimized during training. Common choices for the `criterion` in deep
    learning tasks include functions like `nn.CrossEntropyLoss` for classification tasks or `nn.MSELoss`
    for
    :return: The function `trainMode` returns the trained model (`avm`) after training it on the
    train_dataset for the specified number of epochs and validating it on the val_dataset.
    """
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
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()} Time: {time()-time_}")
            torch.cuda.empty_cache()

        # validation
        print("Validation",end=' ')
        avm = testMode(avm, val_dataset, criterion)
    return avm

def testMode(avm, dataloader, criterion, device):
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
            
        print(f"Loss: {loss.item()}, Accuracy: {acc}")
            
    return avm
    
def splitDataset(dataset):
    """
    The function `splitDataset` takes a dataset and splits it into training, validation, and test sets
    with specified sizes and returns data loaders for each set.
    
    :param dataset: The `dataset` parameter in the `splitDataset` function refers to the dataset that
    you want to split into training, validation, and test sets. This dataset could be any collection of
    data that you want to use for training a machine learning model, such as images, text, or numerical
    data
    :return: The function `splitDataset(dataset)` returns a list containing three DataLoader objects:
    `train_loader`, `val_loader`, and `test_loader`. These DataLoader objects are used to load batches
    of data for training, validation, and testing purposes in machine learning models.
    """
    batch_size = 8
    total_size = len(dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    # Ensure the test set gets any remaining samples after integer division
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return [train_loader, val_loader, test_loader]

# This class `DataLoaderFrameLabeled` is a custom dataset class in PyTorch for loading labeled visual
# frames and audio features.
class DataLoaderFrameLabeled(Dataset):
    def __init__(self, frames,audio, labels):
        self.labels = torch.LongTensor(labels)
        self.visual_frames_tensor = torch.stack([torch.tensor(frame).permute(2, 0, 1) for frame in frames])
        self.audio_features_tensor= torch.tensor(audio)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        visual_frame_tensor = self.visual_frames_tensor[idx]
        audio_feature_tensor =self.audio_features_tensor[idx]
        label = self.labels[idx]

        return visual_frame_tensor, audio_feature_tensor, label


# d240407
# TODO - Dataloader Done
# TODO - Training Loop
 
