import cv2
import numpy as np

def extract_frames(video_path, frame_rate=15):
    video = cv2.VideoCapture(video_path)
    count = 0
    success = True
    frames = []
    
    while success:
        success, image = video.read()
        if count % frame_rate == 0 and success:
            # resize image to 299x299
            image = ((image - image.min()) / (image.max() - image.min())).astype(np.float32)
            image = cv2.resize(image, (299, 299))
            frames.append(image)
            
        count += 1

    video.release()
    return frames