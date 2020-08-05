import numpy as np
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        """
        给定原始的数据集和对应的 index, 产生在 index 中存在的子数据集
        :param dataset:
        :param idxs:
        """
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class MiniDataset(Dataset):

    def __init__(self, data, options):
        """
        这个类在读取的 pkl 为实际的数据的时候用于将 dict 格式的数据转换为 Tensor. 一般进行数据(非index)的预处理
        :param data:
        :param labels:
        """
        super(MiniDataset, self).__init__()
        self.data = np.array(data['x'])
        self.labels = np.array(data['y']).astype(np.int64)
        self.data = self.data.astype(np.float32)
        self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        # if self.data.ndim == 4 and self.data.shape[3] == 3:
        #     data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target