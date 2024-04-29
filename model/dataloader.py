import torch
from torch.utils.data import Dataset

class dataloader(Dataset):

    def __init__(self, frames: list[torch.tensor], audios: list[torch.tensor], labels: list[torch.tensor], device: str = 'cpu'):
        '''
            frames. Dimensions (N, C, H, W).
            audios. Dimensions (N, K, n_mfcc, B).
            labels. Dimensions (N,).

            N is the number of frames; B is the number of timestamps per bin of the MFCC algorithm; C is the number of visual channels.
        '''

        self.frames = [torch.tensor(frames_, dtype = torch.float32, device = device) for frames_ in frames]
        self.audios = [torch.tensor(audios_, dtype = torch.float32, device = device) for audios_ in audios]
        self.labels = [torch.tensor(labels_, dtype = torch.float32, device = device) for labels_ in labels]

        assert len(self.frames) == len(self.audios) == len(self.labels), 'E: Inconsistency in data loader definition'

        self.N = len(self.labels)

    def __len__(self):
        return self.N

    def __getitem__(self, video_idx):
        return self.frames[video_idx], self.audios[video_idx], self.labels[video_idx]
