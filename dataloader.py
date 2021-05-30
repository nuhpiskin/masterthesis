import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as T
import pickle
from PIL import Image
import glob
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class CrackConcrete(Dataset):
    
    def __init__(self,data_path):
        self.images_path = glob.glob(data_path+"/**/*.jpg")
        self.img_transform= T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, id):
        img = self.images_path[id]
        if img.split("/")[-2] == "Negative":
            label = 0
        else:
            label = 1
        y = label
        x = self.img_transform(Image.open(img))
        return x,y