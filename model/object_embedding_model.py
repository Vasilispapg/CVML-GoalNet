import torch
import torch.nn as nn


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


def convert_list_to_dict(class_list):
    return {classname: index for index, classname in enumerate(class_list)}


def process_detected_objects(detected_objects,class_to_idx):
    
    num_classes = len(class_to_idx)  # Number of unique objects
    embedding_dim = 16  # Dimension of the embedding vector
    num_objects_per_frame = 10  # Maximum number of objects detected in a frame
    num_embeddings = num_classes + 1  # Add 1 for the padding token

    # Init the model
    model = ObjectEmbeddingModel(num_embeddings=num_embeddings, embedding_dim=embedding_dim, num_classes=num_classes,num_objects_per_frame=num_objects_per_frame)  # Adjust num_classes based on your task

    # Convert it to a dictionary
    class_to_idx_dict = convert_list_to_dict(class_to_idx) 
    
    # Encode all detected objects
    encoded_batches = []
    for objects in detected_objects:

        encoded_objects = [class_to_idx_dict.get(obj, -1) for obj in objects]
        # Pad or truncate the list of encoded objects to have a fixed size
        encoded_objects = encoded_objects[:num_objects_per_frame]  # Truncate if necessary
        encoded_objects += [0] * (num_objects_per_frame - len(encoded_objects))  # Pad with zeros if necessary
        encoded_batches.append(encoded_objects)
        

    # Ensure the tensor has the correct shape (1, num_detected_objects)
    # If there's a maximum number of objects you want to handle, you might need to pad or truncate the list
    encoded_objects_tensor = torch.LongTensor(encoded_batches)

    # Forward pass
    outputs = model(encoded_objects_tensor)
    # OUTPUT FRAMESx80 (unique coco classes)
    return outputs

# Example how to use the function 
# obj=process_detected_objects([['person', 'car', 'truck','cat'], ['person', 'car'], ['person', 'truck'],[],[]])
