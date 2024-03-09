import pickle
import os
def getData(feature,video):
    if os.path.exists(f'video_ext_data/{video}/{feature}.pkl'):
        with open(f'video_ext_data/{video}/{feature}.pkl', 'rb') as f:
            feature = pickle.load(f)
        return feature
    else:
        return None