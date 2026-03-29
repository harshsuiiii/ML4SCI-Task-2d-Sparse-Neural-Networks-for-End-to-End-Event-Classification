from data.data_loader import H5Dataset
from models.sparse_resnet import SparseResNet
from training.train import train
from utils.helpers import get_device

from torch.utils.data import DataLoader

file_path = "your_path_here"

dataset = H5Dataset(file_path)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = get_device()

model = SparseResNet()
model = train(model, loader, device)