import torch

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
def prepare_frame_features(colors_features, vgg_features, audio_features, objects_encoded):
    """
    Prepares a single frame's features for model input.
    """
    # Assuming colors_features, vgg_features, audio_features, and objects_encoded are already in the correct format
    # and just need to be concatenated.
    breakpoint()
    frame_features = torch.cat((colors_features, vgg_features, audio_features, objects_encoded), 0).unsqueeze(0)  # Add batch dimension
    return frame_features

def evaluate_single_frame(model, frame_features):
    """
    Passes a single frame through the model to get the class probabilities.
    """
    with torch.no_grad():  # No need to track gradients
        outputs = model(frame_features)
        probabilities = outputs.squeeze(0)  # Remove batch dimension
        predicted_class = torch.argmax(probabilities) + 1  # Assuming classes are 1-5
        return probabilities, predicted_class

def evaluate_model(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = correct / total
        print('Test accuracy:', test_acc)

def callNN(visual_features, audio_features, objects_encoded):
    
    # Hyperparameters    
    batch_size = 10
    num_epochs = 2
    
    # Create an instance of the neural network
    model = NeuralNetwork(512+ (3*64) +128 +1024,5)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # FEATURES
    colors_features= [x[0] for x in visual_features]
    vgg_features= [x[1] for x in visual_features]
    
    colors_features = torch.tensor(colors_features)
    vgg_features = torch.tensor(vgg_features)
    audio_features = torch.tensor(audio_features)
    objects_encoded = torch.tensor(objects_encoded)
    
    # CONCATAENATE
    frame_features = prepare_frame_features(colors_features, vgg_features, audio_features, objects_encoded)
    
    train_dataset = data.TensorDataset(frame_features, torch.tensor([1]))
    test_dataset = data.TensorDataset(frame_features, torch.tensor([1]))
    breakpoint()

    # Create the train loader
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the test loader
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    

    # Evaluate the model
    # evaluate_model(model, test_loader)

    # Assuming `model` is your trained model instance
    probabilities, predicted_class = evaluate_single_frame(model, frame_features)
    print(f"Probabilities: {probabilities}")
    print(f"Predicted class: {predicted_class}")