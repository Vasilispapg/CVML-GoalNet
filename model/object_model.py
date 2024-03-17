import torch
import torch.nn as nn
import sys
sys.path.append('DataExtraction')

from objects import loadYOLOv5

class ObjectEmbeddingModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, num_objects_per_frame=10):
        super(ObjectEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # An example linear layer that takes the flattened embeddings as input
        self.fc = nn.Linear(embedding_dim * num_objects_per_frame, num_classes)
    
    def forward(self, x):
        # x is expected to be of shape [batch_size, num_objects_per_frame]
        x = self.embedding(x)  # Output shape: [batch_size, num_objects_per_frame, embedding_dim]
        x = x.view(x.size(0), -1)  # Flatten the embeddings
        x = self.fc(x)
        return x

def encode_objects(objects, class_to_idx):
    # Convert object names to indices
    encoded_objects = [class_to_idx.get(obj, -1) for obj in objects]
    return encoded_objects

def convert_list_to_dict(class_list):
    return {classname: index for index, classname in enumerate(class_list)}


def process_detected_objects(detected_objects_batch):
    # Load YOLOv5 and class indices
    _, class_to_idx = loadYOLOv5()

    num_classes = len(class_to_idx)  # Number of unique objects
    embedding_dim = 16  # Dimension of the embedding vector
    num_objects_per_frame = 10  # Maximum number of objects detected in a frame

    # Init the model
    model = ObjectEmbeddingModel(num_embeddings=num_classes, embedding_dim=embedding_dim, num_classes=num_classes,num_objects_per_frame=num_objects_per_frame)  # Adjust num_classes based on your task

    # Convert it to a dictionary
    class_to_idx_dict = convert_list_to_dict(class_to_idx) 

    # Encode detected objects
    encoded_batches = []
    for objects in detected_objects_batch:
        encoded_objects = encode_objects(objects, class_to_idx_dict)
        # Pad or truncate the list of encoded objects to have a fixed size
        encoded_objects = encoded_objects[:num_objects_per_frame]  # Truncate if necessary
        encoded_objects += [0] * (num_objects_per_frame - len(encoded_objects))  # Pad with zeros if necessary
        encoded_batches.append(encoded_objects)

    encoded_batches_tensor = torch.LongTensor(encoded_batches)  # Convert to tensor

    # Forward pass
    outputs = model(encoded_batches_tensor)
    return outputs

# Example how to use the function 
# obj=process_detected_objects([['person', 'car', 'truck','cat'], ['person', 'car'], ['person', 'truck'],[],[]])
