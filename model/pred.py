import sys
sys.path.append('DataExtraction')
sys.path.append('model')
from model import audio_visual_model
import torch


def pred(audio_features_tensor: list, visual_frames_tensor: list, ):

    N = len(visual_frames_tensor)
# TODO :
# - GET TRAINED MODEL
#  RETURN VALUES : PREDICTIONS
# FINISH THE PRED FUNCTION ON INIT
    avm = audio_visual_model()
    

    predictions = []
    for i in range(N):
        visual_input = visual_frames_tensor[i].unsqueeze(0)
        audio_input = audio_features_tensor[i].unsqueeze(0)
        prediction = avm(audio_input = audio_input, visual_input = visual_input)
        predictions.append(round(prediction.item()))
        torch.cuda.empty_cache()
    print(predictions)
