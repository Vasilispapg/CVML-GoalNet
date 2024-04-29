import cv2
import numpy as np
import torch

def extract_visual_frames(fp: str, skip_frames: int = 15):

    video = cv2.VideoCapture(fp)
    count = 0
    success = True
    frames = []

    while success:
        success, image = video.read()
        if count % skip_frames == 0 and success:
            image = ((image - image.min()) / (image.max() - image.min() + 1e-7)).astype(np.float32)
            image = cv2.resize(image, (299, 299))
            frames.append(image)
        count += 1

    video.release()

    return np.transpose(np.array(frames), axes = (0, 3, 1, 2))


def get_frame_numbers(encoded_frames, hdf5_file):
    frame_numbers = []
    for ref_array in encoded_frames:
        for ref in ref_array:
            frame_data = hdf5_file[ref]
            frame_numbers.extend([int(char[0]) for char in frame_data])
    return frame_numbers


def get_visual_features(model, sample_visual_frames):
    model.eval()
    visual_descriptor_tensor = []
    for frame in sample_visual_frames:
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        output = model(frame)
        visual_descriptor_tensor.append(output)

    return visual_descriptor_tensor