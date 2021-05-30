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
def train(args):
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    writer = SummaryWriter(args.save_folder+args.exp_name)
    num_classes = 2
    num_workers = args.num_workers
    training_dataset = args.training_dataset
    save_folder = args.save_folder+args.exp_name+"/"

    train_dataset = CrackConcrete("/media/syn/7CC4B2EE04A2CEAE/private/Crack_classification")
    test_dataset = CrackConcrete("/media/syn/7CC4B2EE04A2CEAE/private/Crack_classification_Val")

    train_loader = DataLoader(dataset= train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=8)
    test_loader = DataLoader(dataset= test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=8)
    device = "cuda"
    net = CrackClassificationModels(model_name = args.network,num_classes = num_classes)
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
    for e in tqdm.trange(1, args.epochs+1):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc_mask = 0
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch = X_train_batch.cuda()
            y_train_batch = y_train_batch.cuda()
            optimizer.zero_grad()
            y_train_pred_mask = net(X_train_batch)
            train_loss_mask = criterion(y_train_pred_mask,y_train_batch)
            train_loss = train_loss_mask
            train_acc_mask = multi_acc(y_train_pred_mask, y_train_batch)
            
            train_loss.backward()
            optimizer.step()


            writer.add_scalar("Loss/train", train_loss, counter)
            writer.add_scalar("Accuracy/train_mask", train_acc_mask.item(), counter)
            counter += 1
        if e%5==0:
            # VALIDATION    
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

                writer.add_scalar("Loss/test", val_epoch_loss/len(test_loader), e)
                writer.add_scalar("Accuracy/test_mask", val_epoch_acc_mask/len(test_loader), e)

        if e%10==0:
            if best<val_epoch_loss/len(test_loader):

                save_checkpoint(net.state_dict(),is_best=True,epoch = e)
                best=val_epoch_loss/len(test_loader)
            else:
                save_checkpoint(net.state_dict(),is_best=False,epoch = e)

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
    parser.add_argument('--validation_nms', default=0.4, help='Validation non maxima threshold')
    parser.add_argument('--validation_th', default=0.02, help='Validation confidence threshold')
    parser.add_argument('--batch_size', default=16, type=int, help='Validation confidence threshold')
    parser.add_argument('--epochs', default=100, type=int, help='Validation confidence threshold')
    args = parser.parse_args()

    for network in ["swin_transformer", "efficentnet","resnet50","resnet18"]:
        args.network = network
        args.exp_name = args.network + "_iter1"
        train(args)