from models.resnet18_unet import *
from collections import defaultdict
import torch.nn.functional as F
from utils.dataloader import CrackConcrete, CrackConcreteData
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.utils.data as data
import torch.nn as nn
import time
import datetime
import math
import shutil
from tqdm import trange
from torchvision.ops import nms
import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.loss import *
from torch.optim import lr_scheduler
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from timer import Timer
from sklearn.metrics import jaccard_similarity_score
import numpy as np

def iou(pred, target, n_classes = 2):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious)


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    # metrics['iou'] = iou(target,pred)

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    metrics['jaccard'] += jaccard_similarity_score(target_np.ravel(), pred_np.ravel())
    
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs))) 

def val(args): 


    num_classes = 1
    batch_size = args.batch_size


    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])
    train_set = CrackConcreteData(datapath = "/media/syn/7CC4B2EE04A2CEAE/Thesis/Segmentation/Train" ,transform=trans)
    val_set = CrackConcreteData(datapath = "/media/syn/7CC4B2EE04A2CEAE/Thesis/Segmentation/Val",transform=trans)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8)
    }

    device = "cuda"
    net = ResNetUNet(num_classes)
    net = net.to(device)

    for l in net.base_layers:
        for param in l.parameters():
            param.requires_grad = False
    
    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = torch.load("logs/"+args.resume_net+f"/checkpoint_{i}.pth")
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    net = net.cuda()
    net.eval()
    phase = "val"
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1) 

        
    since = time.time()

    net.eval()   # Set model to evaluate mode

    metrics = defaultdict(float)
    epoch_samples = 0
    
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)             

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            sy = Timer.start()
            outputs = net(inputs)
            Timer.end(sy, "Model Inference Time", level="1")
            loss = calc_loss(outputs, labels, metrics)


        # statistics
        epoch_samples += inputs.size(0)

    print_metrics(metrics, epoch_samples, phase)
    epoch_loss = metrics['loss'] / epoch_samples
    

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
    print('Best val loss: {:4f}'.format(epoch_loss))
    Timer.print_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument('--training_dataset', default='/media/syn/7CC4B2EE04A2CEAE/private/Crack_classification/', help='Training dataset directory')
    parser.add_argument('--network', default='swin_transformer', help='Backbone network resnet18, resnet50,efficentnet, swin_transformer')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_folder', default='./logs/', help='Location to save checkpoint models')
    parser.add_argument('--exp_name', default='test1', help='Location to save checkpoint models')
    parser.add_argument('--batch_size', default=4, type=int, help='Validation confidence threshold')
    parser.add_argument('--epochs', default=100, type=int, help='Validation confidence threshold')
    args = parser.parse_args()
    args.resume_net = "test1"
    for i in ["0","4","8","16","24","60"]:
        val(args)