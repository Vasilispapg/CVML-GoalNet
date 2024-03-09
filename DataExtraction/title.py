import numpy as np
import pandas as pd

def tfTitle(info_file,video_path,tokenizer):
    info_df = pd.read_csv(info_file, sep='\t')
    info_video=info_df[info_df['video_id']==video_path.split('/')[-1].split('.')[0]]
    title=info_video['title'].values[0]

    print('title:',title)

    title_features = tokenizer(title).vector
    # print('title_features:',title_features.shape)
    return np.array(title_features)