import h5py
import torch
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None

    def _open(self):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')
            self.X = self.file["jet"]
            self.y = self.file["Y"]

    def __len__(self):
        self._open()
        return len(self.X)

    def __getitem__(self, idx):
        self._open()

        x = self.X[idx].astype('float32')
        x = (x - x.mean()) / (x.std() + 1e-8)

        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return x, y