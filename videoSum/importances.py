import numpy as np

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(np.array(vec1) - np.array(vec2)))

from sklearn.metrics.pairwise import cosine_similarity

def cosine_distance(vec1, vec2):
    # Ensure the vectors are 2D and of the same length
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)

    # Check if either vector contains only NaNs
    if np.isnan(vec1).all() or np.isnan(vec2).all():
        return 0  # or handle as needed

    return cosine_similarity(vec1, vec2)[0][0]

def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)

    # Avoid division by zero in case all scores are the same
    if max_score == min_score:
        return np.zeros_like(scores)

    normalized_scores = (scores - min_score) / (max_score - min_score)
    return normalized_scores

def calculate_importance(title, objects):
    importance_scores = []

    for frame_objects in objects:
        frame_distance = 0

        for obj in frame_objects:
            obj_array = np.array(obj)
            title_array = np.array(title)

            # Calculate distance and sum it up for the frame
            distance = cosine_distance(obj_array, title_array)
            frame_distance += distance

        # Store the summed distance for the frame
        importance_scores.append(frame_distance)

    # importance_scores=normalize_scores(importance_scores)
    importance_scores = np.array(importance_scores)
        
    # Invert the scores because lower distance indicates higher similarity
    importance_scores = 1 - importance_scores
        
    # Handle zero scores to avoid division by zero later
    for i in range(len(importance_scores)):
        if importance_scores[i] == 0:
            importance_scores[i] = 0.0001
            
    return importance_scores