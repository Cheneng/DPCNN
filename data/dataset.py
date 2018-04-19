from torch.utils import data
import os


class TextDataset(data.Dataset):

    def __init__(self, path):
        self.file_name = os.listdir(path)

    def __getitem__(self, index):
        return self.train_set[index], self.labels[index]

    def __len__(self):
        return len(self.train_set)


