import cv2

def create_video_from_frames(frames, output_path, frame_rate=30):
    if not frames:
        print("No frames to create a video.")
        return None
    # Determine the width and height from the first frame
    height, width, layers = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    
    # Write each frame to the video
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()
    return output_path