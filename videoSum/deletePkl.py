import os
def deletePKLfiles(video):
    dirpkl='video_ext_data/'+video+'/'
    if(os.path.exists(dirpkl+'frames.pkl')):
        os.remove(dirpkl+"frames.pkl")
    return
