import cv2
import numpy as np
from PIL import Image


def detect_objects_from_frames(frames, cluster_labels, model, classes):
    """
    Detect unique objects in video frames grouped by cluster labels.
    
    Args:
    frames: List of video frames.
    cluster_labels: Cluster labels for each frame.
    model: The loaded YOLO model.
    classes: A list of class names that YOLO can detect.

    Returns:
    A dictionary of unique detected objects grouped by cluster label.
    """
    print("Object detection starts")
    
    cluster_labels = cluster_labels.flatten()  # This converts a 2D array to a 1D array
   
    # set of unique cluster labels
    clustered_frames = {label: [] for label in set(cluster_labels)}
    
    # attach each frame in the video to its corresponding cluster label
    for label, frame in zip(cluster_labels, frames):
        # Add 50 frames to each cluster label
        for _ in range(100):
            # Append the frame to the list of frames for the corresponding cluster label
            clustered_frames[label].append(frame)

    # detect objects in each frame of the video
    detected_objects = {label: set() for label in clustered_frames}
    for label, frames in clustered_frames.items():
        for frame in frames:
            detected_labels = detect_objects_with_yolo(frame, model, classes)
            for obj in detected_labels:
                detected_objects[label].add(obj)

    return {label: list(objects) for label, objects in detected_objects.items()}

def detect_objects_in_all_frames(frames, model, classes):
    """
    Detect objects in all video frames.

    Args:
    frames: List of video frames.
    model: The loaded YOLO model.
    classes: A list of class names that YOLO can detect.

    Returns:
    A list where each element contains the detected objects in the corresponding frame.
    """
    print("Object detection on each frame starts")
    
    detected_objects_per_frame = []

    # Perform object detection on each frame
    for frame in frames:
        detected_objects = detect_objects_with_yolo(frame, model, classes)
        detected_objects_per_frame.append(detected_objects)
        
    print("Object detection on each frame ends")

    return detected_objects_per_frame

def detect_objects_with_yolo(frame, model, classes):
    """
    Detect objects in a frame using the YOLO model.

    Args:
    frame: A single frame to perform object detection on.
    model: The loaded YOLO model.
    classes: A list of class names that YOLO can detect.

    Returns:
    A list of detected object labels in the frame.
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform inference
    results = model(pil_image)

    # Parse results
    detected_labels = set()
    for _, _, _, _, conf, cls_id in results.xyxy[0].cpu().numpy():
        if conf > 0.85:  # Confidence threshold
            detected_labels.add(classes[int(cls_id)])

    return list(detected_labels)

