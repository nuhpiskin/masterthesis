import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
import torchvision
import glob
import random
from model import AE
import pickle
from torch.utils.tensorboard import SummaryWriter
import os

class Dataloader:
    def __init__(self,datas):
        self.train = datas
    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        data  =self.train[idx]
        data = pd.read_csv(data, encoding = "ISO-8859-1", skiprows=11,delim_whitespace =True,header = None)
        out = data.loc[:,0].values[None].astype("float32")
        out = (out-out.min())/(out.max()-out.min())
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=2048).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()
writer = SummaryWriter("./logs")

# path = "/home/nmp/Desktop/Tez/masterthesis/data/nuh"
# datas = glob.glob(path+"/**/*.csv")
# print("Total Data Size:",len(datas))
# random.shuffle(datas)
# test_datas = datas[int(len(datas)*0.9):]
# train_datas = datas[:int(len(datas)*0.9)]

with open("data.pkl","rb") as f:
    datas = pickle.load(f)

test_datas = datas["Train"]
train_datas = datas["Test"]

print("Test Datas::",len(test_datas))
print("Train Datas:",len(train_datas))


train_dataset = Dataloader(train_datas)

test_dataset = Dataloader(test_datas)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False
)
epochs = 100
c = 0
for epoch in range(epochs):
    loss = 0
    for batch_features in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.to(device)
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs = model(batch_features)
        
        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
    # compute the epoch training loss
    loss = loss / len(train_loader)
    writer.add_scalar("Loss/train", loss,epoch)
    if epoch%5 == 0:
        torch.save(model.state_dict(),os.path.join("./logs", f"epoch{epoch}.pth"))
    if epoch %10 == 0:
        loss_test = 0
        for batch_features in test_loader:
            batch_features = batch_features.to(device)
        
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training reconstruction loss
            test_loss = criterion(outputs, batch_features)
            

            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss_test += test_loss.item()
            
    # compute the epoch training loss
    loss_test = loss_test / len(test_loader)
    writer.add_scalar("Loss/test", loss_test,epoch)
