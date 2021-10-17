import os
from numpy.lib.type_check import imag
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as T
import pickle
from PIL import Image
import glob
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from utils import simulation
import cv2
class CrackConcrete(Dataset):
    def __init__(self, count, transform=None):
        
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)        
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
        
        return [image, mask]

class CrackConcreteData(Dataset):
    def __init__(self, datapath, transform=None):
        self.datapath = datapath 
        self.input_images = os.listdir(os.path.join(datapath, "Images"))
        self.target_masks = os.listdir(os.path.join(datapath, "Masks"))
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image_path = self.input_images[idx]
        mask_path = self.target_masks[idx]
        image = cv2.imread(os.path.join(self.datapath,"Images",image_path))
        mask = cv2.imread(os.path.join(self.datapath,"Masks",mask_path))
        size = (192,192)
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),size)
        mask[mask!=0] = 1
        mask = cv2.resize(cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY), size)
        mask = mask[None].transpose(2,1,0)
        if self.transform:
            image = self.transform(image)
        
        return [image, mask]