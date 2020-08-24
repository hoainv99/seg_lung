from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
import cv2 
import numpy as np
import torch
import torch.nn.functional as F
class MyDataset(Dataset):
    def __init__(self,data_dict, trans=None):
        super(MyDataset).__init__()
        self.data_dict = data_dict
        self.trans = trans

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self,idx):
        image_path = self.data_dict[idx]['image_path']
        label_path = self.data_dict[idx]['label_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(np.uint8(image))
        if self.trans:
            image = self.trans(image)
            
        return image, torch.from_numpy(label).float()
        