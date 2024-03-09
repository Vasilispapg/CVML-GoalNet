import cv2

def extract_frames(video_path, frame_rate=15):
    video = cv2.VideoCapture(video_path)
    count = 0
    success = True
    frames = []
    
    while success:
        success, image = video.read()
        if count % frame_rate == 0 and success:
            frames.append(image)
        count += 1

    video.release()
    return frames