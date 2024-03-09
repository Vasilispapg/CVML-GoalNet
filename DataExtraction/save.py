import pickle
import os
def saveData(name,feature,video):
    if not os.path.exists(f'video_ext_data/{video}'):
        os.makedirs(f'video_ext_data/{video}')
    with open(f'video_ext_data/{video}/{name}.pkl', 'wb') as f:
        pickle.dump(feature,f)