
from mapping import getChangingPoints
from mapping import video_id_map

def getClipImportances(importance, video):
    
    # Get the changing points for the video
    changingPoints = getChangingPoints(video_id_map[video.split('.')[0]])
        
    # Initialize a dictionary to store clip importances
    clip_importances = {}

    # Iterate over each clip defined by changing points
    for clip_index, (start_frame, end_frame) in enumerate(changingPoints):
        clip_importance = 0
        num_frames=end_frame-start_frame

        # Calculate the total importance for this clip
        for i in range(start_frame, end_frame):
            clip_importance += importance[i]

        # Store the clip importance
        clip_importances[clip_index] = (clip_importance,num_frames)
        
    # normalize
    return clip_importances

def getSelectedIndicesFromClips(selectedClips,video):
    changingPoints = getChangingPoints(video_id_map[video.split('.')[0]])

    # Initialize a list to store the selected indices
    selected_indices = []
    for i in selectedClips:
        selected_indices.extend(range(changingPoints[i][0], changingPoints[i][1]))
    return selected_indices