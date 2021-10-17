from models.resnet18_unet import *
from collections import defaultdict
import torch.nn.functional as F
from dataloader import CrackConcrete
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
from loss import *
from torch.optim import lr_scheduler
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs))) 



def train(args):

    def save_checkpoint(state, is_best,epoch,filename=f'/logs/{args.exp_name}/checkpoint.pth'):
        torch.save(state, "."+filename.split(".")[0]+"_"+str(epoch)+".pth")
        if is_best:
            shutil.copyfile("."+filename.split(".")[0]+"_"+str(epoch)+".pth", os.path.join("./logs",args.exp_name,'model_best.pth'))

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    writer = SummaryWriter(args.save_folder+args.exp_name)
    num_classes = 6
    batch_size = args.batch_size
    num_epochs = args.epochs
    num_workers = args.num_workers
    training_dataset = args.training_dataset
    save_folder = args.save_folder+args.exp_name+"/"

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])
    train_set = CrackConcrete(2000, transform=trans)
    val_set = CrackConcrete(200, transform=trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    device = "cuda"
    net = ResNetUNet(num_classes)
    net = net.to(device)

    for l in net.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1) 
    

    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = torch.load(args.resume_net)
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
    net.train()
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 1e10

    for epoch in tqdm.trange(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                exp_lr_scheduler.step()
                for param_group in optimizer_ft.param_groups:
                    print("LR", param_group['lr'])
                    
                net.train()  # Set model to training mode
            else:
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
                    outputs = net(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if epoch%4 == 0:
                save_checkpoint(net.state_dict(),is_best=False, epoch=epoch)
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
                save_checkpoint(net.state_dict(),is_best=True, epoch=epoch)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
    print('Best val loss: {:4f}'.format(best_loss))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument('--training_dataset', default='/media/syn/7CC4B2EE04A2CEAE/private/Crack_classification/', help='Training dataset directory')
    parser.add_argument('--network', default='swin_transformer', help='Backbone network resnet18, resnet50,efficentnet, swin_transformer')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_folder', default='./logs/', help='Location to save checkpoint models')
    parser.add_argument('--exp_name', default='debug', help='Location to save checkpoint models')
    parser.add_argument('--batch_size', default=1, type=int, help='Validation confidence threshold')
    parser.add_argument('--epochs', default=100, type=int, help='Validation confidence threshold')
    args = parser.parse_args()
    
    train(args)