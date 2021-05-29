from pathlib import Path

import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data

import datahandler
from model import createDeepLabv3
from train import train_model
import cv2
from torchvision import transforms
import numpy as np
def image():
    data_transforms = transforms.Compose([transforms.ToTensor()])
    img_org = cv2.imread("/home/syn/Desktop/private/masterthesis/images/test.jpg")
    img_org = cv2.resize(img_org,(960,540))
    img_org = cv2.cvtColor(img_org,cv2.COLOR_BGR2RGB)
    img = data_transforms(img_org)
    outputs = model(img[None].cuda())
    y_pred = outputs['out'].data.cpu().numpy()
    pixels = np.where(y_pred[0][0]>0.2)
    tmp_mask = np.zeros((y_pred[0][0].shape[0],y_pred[0][0].shape[1],3),dtype = np.uint8)
    tmp_mask[pixels[0],pixels[1],0] = 50
    tmp_mask[pixels[0],pixels[1],1] = 124
    tmp_mask[pixels[0],pixels[1],2] = 232
    alpha = 0.45
    img_org = cv2.addWeighted(tmp_mask, alpha, img_org, 1 - alpha,
		0, 0)
    cv2.imwrite("test.png",img_org[:,:,::-1])

if __name__ == '__main__':
    model = torch.load("/home/syn/Desktop/private/masterthesis/logs/weights.pt")
    image()