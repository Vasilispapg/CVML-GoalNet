import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
# dn eimai sigoyros
from xception import xception
import numpy as np

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
        audio_features = torch.flatten(audio_features)[np.newaxis, ...]
        # VISUAL FEATURES
        xception_features = self.visual_model(visual_input)  # Extract features using Xception
        visual_features = self.visual_model_expand(xception_features)  # Further process features
        visual_features = torch.flatten(visual_features, start_dim=1)  # Flatten the features to shape: [batch_size, feature_dim_visual]
        combined_features = torch.cat((audio_features, visual_features), axis = -1)
        output = self.fusion(combined_features)
        output = F.softmax(output, dim=1)
        return output
    
def proccessForEachFrame(sample_visual_frames, avm, visual_frames_tensor, audio_features_tensor):
     # Process each frame
    predictions = []
    for i in range(len(sample_visual_frames)):
        visual_input = visual_frames_tensor[i].unsqueeze(0)  # Add batch dimension
        audio_input = audio_features_tensor[i].unsqueeze(0).T  # Add batch dimension
        prediction = avm(audio_input, visual_input)
        class_idx = torch.argmax(prediction)
        # [0, 1, 2, 3, 4] -> [1, 2, 3, 4, 5] 
        predictions.append(class_idx.item() + 1)
        torch.cuda.empty_cache()
    return [avm, predictions]

def callNN(sample_visual_frames, audio_features, labels):
    
    # Hyperparameters    
    batch_size = 10
    num_epochs = 2
    
    # Initialize your model
    # Refine descriptors - avm: Audio visual model
    avm = AudioVisualModel()
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(avm.parameters())
    
    # Convert frames and audio features to PyTorch tensors
    visual_frames_tensor = torch.stack([torch.tensor(frame).permute(2, 0, 1) for frame in sample_visual_frames])
    audio_features_tensor = torch.tensor(audio_features)
    avm,predictions=proccessForEachFrame(sample_visual_frames, avm, visual_frames_tensor, audio_features_tensor)
    print(predictions)
    breakpoint()
    

 
