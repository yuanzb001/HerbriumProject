import os.path
import numpy as np
from PIL import Image
from sklearn import preprocessing
import torch

from torch.utils.data import Dataset

class myDataSet(Dataset):
    def __init__(self, m_fold_path, m_img_path_list, m_label_list, m_img_transform = None):
        le = preprocessing.LabelEncoder()
        self.img_list = [os.path.join(m_fold_path, filename) for filename in m_img_path_list]
        le.fit(m_label_list)
        self.label_list = le.transform(m_label_list)
        self.img_transform = m_img_transform
        self.classes = np.unique(m_label_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        label = torch.as_tensor(self.label_list[item])
        img = Image.open(img_path).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return len(self.label_list)