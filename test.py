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
from models.crack_models import CrackClassificationModels
from torch.utils.tensorboard import SummaryWriter
def test(args):
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    writer = SummaryWriter(args.save_folder+args.exp_name)
    num_classes = 2
    num_workers = args.num_workers
    training_dataset = args.training_dataset
    save_folder = args.save_folder+args.exp_name+"/"

    train_dataset = CrackConcrete("/media/nmp/C/Tez/dataset/Classification/Train/Train")
    test_dataset = CrackConcrete("/media/nmp/C/Tez/dataset/Classification/Train/Val")

    train_loader = DataLoader(dataset= train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=8)
    test_loader = DataLoader(dataset= test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=8)
    device = "cuda"
    net = CrackClassificationModels(model_name = args.network,num_classes = num_classes)
    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = torch.load("weights/swin/model_best.pth")
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
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    def multi_acc(y_pred, y_test):
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
        
        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)
        
        acc = acc * 100
        
        return acc

    def save_checkpoint(state, is_best,epoch,filename=f'/logs/{args.exp_name}/checkpoint.pth'):
        torch.save(state, "."+filename.split(".")[0]+"_"+str(epoch)+".pth")
        if is_best:
            shutil.copyfile("."+filename.split(".")[0]+"_"+str(epoch)+".pth", os.path.join("./logs",args.exp_name,'model_best.pth'))

    best = 0
    counter = 0
    with torch.no_grad():
                
        val_epoch_loss = 0
        val_epoch_acc_mask = 0

    net.eval()
    for X_val_batch, y_val_batch in test_loader:
        X_val_batch = X_val_batch.cuda()
        y_val_batch = y_val_batch.cuda()

        y_val_pred_mask = net(X_val_batch)
        val_loss_mask = criterion(y_val_pred_mask, y_val_batch)

        val_loss = val_loss_mask
        val_acc_mask = multi_acc(y_val_pred_mask, y_val_batch)

        val_epoch_loss += val_loss.item()
        val_epoch_acc_mask += val_acc_mask.item()
    print("Validation Loss",val_epoch_loss/len(test_loader))
    print("Accuracy",val_epoch_acc_mask/len(test_loader))
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument('--training_dataset', default='/media/syn/7CC4B2EE04A2CEAE/private/Crack_classification/', help='Training dataset directory')
    parser.add_argument('--network', default='swin_transformer', help='Backbone network resnet18, resnet50,efficentnet, swin_transformer')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--resume_net', default="true", help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_folder', default='./logs/', help='Location to save checkpoint models')
    parser.add_argument('--exp_name', default='debug', help='Location to save checkpoint models')
    parser.add_argument('--validation_nms', default=0.4, help='Validation non maxima threshold')
    parser.add_argument('--validation_th', default=0.02, help='Validation confidence threshold')
    parser.add_argument('--batch_size', default=2, type=int, help='Validation confidence threshold')
    parser.add_argument('--epochs', default=50, type=int, help='Validation confidence threshold')
    args = parser.parse_args()

    args.network = ["efficentnet","resnet50","resnet18","swin_transformer"][3]
    args.exp_name = args.network + "_iter1"
    test(args)