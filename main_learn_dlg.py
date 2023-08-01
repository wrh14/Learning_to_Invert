import argparse
import numpy as np
from tqdm import tqdm
import math
from copy import deepcopy
import os
os.environ['KMP_WARNINGS'] = '0'
# import cPickle as pickle
import pickle
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from scipy.optimize import linear_sum_assignment
from scipy.fftpack import dct, idct

from utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import LeNetMnist, weights_init, LeNet
from models.resnet import resnet20
from logger import set_logger

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='dataset to do the experiment')
parser.add_argument('--model', type=str, default="MLP-3000",
                    help='MLP-{hidden_size}')
parser.add_argument('--shared_model', type=str, default="LeNet",
                    help='LeNet')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='epochs for training')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size for training')
parser.add_argument('--leak_mode', type=str, default="sign",
                    help='sign/prune-{prune_rate}/batch-{batch_size}')
parser.add_argument('--trainset', type=str, default="full")
args = parser.parse_args()
logger = set_logger("", f"print_logs/{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.txt")
logger.info(args)

def train(grad_to_img_net, data_loader, sign=False, mask=None, prune_rate=None, leak_batch=1):
    grad_to_img_net.train()
    total_loss = 0
    total_num = 0
    for i, (xs, ys) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        batch_num = len(ys)
        batch_size = int(batch_num / leak_batch)
        batch_num = batch_size * leak_batch
        total_num += batch_num
        xs, ys = xs[:batch_num, selected_para].cuda(), ys[:batch_num].cuda()
        if sign:
            xs = torch.sign(xs)
        if prune_rate is not None:
            mask = torch.zeros(xs.size()).cuda()
            rank = torch.argsort(xs.abs(), dim=1)[:,  -int(xs.size()[1] * (1 - prune_rate)):]
            mask[torch.arange(len(ys)).view(-1, 1).expand(rank.size()), rank] = 1   
        if mask is not None:
            xs = xs * mask
        if gauss_noise > 0:
            xs = xs + torch.randn(*xs.shape).cuda() * gauss_noise
        xs = xs.view(batch_size, leak_batch, -1).mean(1)
        ys = ys.view(batch_size, leak_batch, -1)
        preds = grad_to_img_net(xs).view(batch_size, leak_batch, -1)
        
        batch_wise_mse = (torch.cdist(ys, preds) ** 2) / image_size
        loss = 0
        for mse_mat in batch_wise_mse:
            row_ind, col_ind = linear_sum_assignment(mse_mat.detach().cpu().numpy())
            loss += mse_mat[row_ind, col_ind].mean()

        loss /= batch_size
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_num
            
    total_loss = total_loss / len(data_loader.dataset)
    return total_loss


def test(grad_to_img_net, data_loader, sign=False, mask=None, prune_rate=None, leak_batch=1):
    grad_to_img_net.eval()
    total_loss = 0
    total_num = 0
    reconstructed_data = []
    with torch.no_grad():
        for i, (xs, ys) in enumerate(tqdm(data_loader)):
            batch_num = len(ys)
            batch_size = int(batch_num / leak_batch)
            batch_num = batch_size * leak_batch
            total_num += batch_num
            xs, ys = xs[:batch_num, selected_para].cuda(), ys[:batch_num].cuda()
            if sign:
                xs = torch.sign(xs)
            if prune_rate is not None:
                mask = torch.zeros(xs.size()).cuda()
                rank = torch.argsort(xs.abs(), dim=1)[:,  -int(xs.size()[1] * (1 - prune_rate)):]
                mask[torch.arange(len(ys)).view(-1, 1).expand(rank.size()), rank] = 1   
            if mask is not None:
                xs = xs * mask
            if gauss_noise > 0:
                xs = xs + torch.randn(*xs.shape).cuda() * gauss_noise
            xs = xs.view(batch_size, leak_batch, -1).mean(1)
            ys = ys.view(batch_size, leak_batch, -1)
            preds = grad_to_img_net(xs).view(batch_size, leak_batch, -1)
            batch_wise_mse = (torch.cdist(ys, preds) ** 2) / image_size
            loss = 0
            for batch_id, mse_mat in enumerate(batch_wise_mse):
                row_ind, col_ind = linear_sum_assignment(mse_mat.detach().cpu().numpy())
                loss += mse_mat[row_ind, col_ind].sum()
                #save the reconstructed data in order
                sorted_preds = preds[batch_id, col_ind]
                sorted_preds[row_ind] = preds[batch_id, col_ind]
                sorted_preds = sorted_preds.view(leak_batch, -1).detach().cpu()
                reconstructed_data.append(sorted_preds)
            total_loss += loss.item()
            
    reconstructed_data = torch.cat(reconstructed_data)
    reconstructed_data = reconstructed_data.view(-1, 3, 32, 32)
    total_loss = total_loss / total_num
    return total_loss, reconstructed_data

#input the model shared among parties
image_size = 3 * 32 * 32
num_classes = 10

if args.shared_model == "LeNet":
    net = LeNet(num_classes).to("cuda")
    compress_rate = 1.0
    torch.manual_seed(1234)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot
model_size = 0
for i, parameters in enumerate(net.parameters()):
    model_size += np.prod(parameters.size())
logger.info(f"model size: {model_size}")

# net.load_state_dict(torch.load("cifar_lenet.pth"))
#generate training / test dataset
if args.trainset == "full":
    checkpoint_name = f"data/{args.dataset}_{args.shared_model}_grad_to_img.pl"
else:
    checkpoint_name = f"data/{args.dataset}_{args.shared_model}_{args.trainset}_grad_to_img.pl"
if not os.path.exists(checkpoint_name):
    print("generating dataset...")
    if args.dataset == "MNIST":
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        dst_train = datasets.MNIST("~/.torch", download=True, train=True, transform=transform)
        dst_test = datasets.MNIST("~/.torch", download=True, train=False, transform=transform)
    elif args.dataset == "FashionMNIST":
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        dst_train = datasets.FashionMNIST("~/.torch", download=True, train=True, transform=transform)
        dst_test = datasets.FashionMNIST("~/.torch", download=True, train=False, transform=transform)
    elif args.dataset == "CIFAR10":
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        dst_train = datasets.CIFAR10("~/.torch", download=True, train=True, transform=transform)
        dst_test = datasets.CIFAR10("~/.torch", download=True, train=False, transform=transform)    

    if args.trainset == "ood":
        torch.manual_seed(12345)
        dst_train.targets = np.asarray(dst_train.targets)
        selection_2 = np.arange(len(dst_train.targets))[np.any(np.concatenate([np.expand_dims(dst_train.targets == i, 0) for i in range(5, 10)], 0), axis=0)]
        img_size = dst_train.data.shape[1]
        dct_train = dct(dct(dst_train.data[selection_2], axis=2, norm="ortho"), axis=3, norm="ortho")

        mean_train_prior = dct_train.mean(0)
        std_train_prior = dct_train.std(0)

        x_dct = np.random.randn(len(selection_2), img_size, img_size, 3) * dct_train.std(0) + dct_train.mean(0)
        x_idct = idct(idct(x_dct, axis=3, norm='ortho'), axis=2, norm='ortho')
        pseudo_data = x_idct.astype(np.uint8)    

        dst_train.data, dst_train.targets = np.concatenate([dst_train.data[selection_2], pseudo_data]), np.concatenate([dst_train.targets[selection_2],  np.random.randint(0, 10, len(selection_2))])

        dst_test.targets = np.asarray(dst_test.targets)
        selection_1 = np.arange(len(dst_test.targets))[np.any(np.concatenate([np.expand_dims(dst_test.targets == i, 0) for i in range(5)], 0), axis=0)]
        dst_test.data, dst_test.targets = dst_test.data[selection_1], dst_test.targets[selection_1]

    train_loader = torch.utils.data.DataLoader(dataset=dst_train,
                                                  batch_size=1, 
                                                  shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dst_test,
                                                  batch_size=1, 
                                                  shuffle=False)
    def leakage_dataset(data_loader):
        targets = torch.zeros([len(data_loader.dataset), image_size])
        features = torch.zeros([len(data_loader.dataset), model_size])

        for i, (images, labels) in enumerate(tqdm(data_loader)):
            onehot_labels = label_to_onehot(labels, num_classes=num_classes)
            images, onehot_labels = images.cuda(), onehot_labels.cuda()
            pred = net(images)
            loss = criterion(pred, onehot_labels)
            dy_dx = torch.autograd.grad(loss, net.parameters())
            original_dy_dx = torch.cat(list((_.detach().clone().view(-1) for _ in dy_dx)))
            targets[i] = images.view(-1)
            features[i] = original_dy_dx
        return features, targets
    #parallel saving
    checkpoint = {}
    features, targets = leakage_dataset(train_loader)
    checkpoint["train_features"] = features
    checkpoint["train_targets"] = targets
    features, targets = leakage_dataset(test_loader)
    checkpoint["test_features"] = features
    checkpoint["test_targets"] = targets
    torch.save(checkpoint, checkpoint_name)
else:
    checkpoint = torch.load(checkpoint_name)
del net
    
    
print("loading dataset...")
trainset = torch.utils.data.TensorDataset(checkpoint["train_features"], checkpoint["train_targets"])
testset = torch.utils.data.TensorDataset(checkpoint["test_features"], checkpoint["test_targets"])

#leakage mode
prune_rate = None
leak_batch = 1
sign = False
gauss_noise = 0
leak_mode_list = args.leak_mode.split("-")
for i in range(len(leak_mode_list)):
    if leak_mode_list[i] == "sign":
        sign = True
    elif leak_mode_list[i] == "prune":
        prune_rate = float(leak_mode_list[i+1])
    elif leak_mode_list[i] == "batch":
        leak_batch = int(leak_mode_list[i+1])
    elif leak_mode_list[i] == "gauss":
        gauss_noise = float(leak_mode_list[i+1])
print(prune_rate, leak_batch, sign, gauss_noise)

#init the model
torch.manual_seed(0)
selected_para = torch.randperm(model_size)[:int(model_size * compress_rate)]
if args.model.startswith("MLP"):
    print(image_size)
    hidden_size = int(args.model.split("-")[-1])
    grad_to_img_net = nn.Sequential(
        nn.Linear(len(selected_para), hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, image_size * leak_batch),
        torch.nn.Sigmoid()
    )
    grad_to_img_net = grad_to_img_net.cuda()
    
size = 0
for parameters in grad_to_img_net.parameters():
    size += np.prod(parameters.size())
print(f"net size: {size}")

#training set-up
lr = args.lr
epochs = args.epochs
optimizer = torch.optim.Adam(grad_to_img_net.parameters(), lr=lr)
    
#load the dataloader
batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=(batch_size * leak_batch), 
                                              shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=(batch_size * leak_batch), 
                                              shuffle=False)
#reformate the gt_data
gt_data = checkpoint["test_targets"]
gt_data = gt_data.view(-1, 3, 32, 32)
del checkpoint

#learning
if args.trainset == "full":
    checkpoint_name = f"checkpoint/{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.pt"
else:
    checkpoint_name = f"checkpoint/{args.dataset}_{args.trainset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.pt"
    
if os.path.exists(checkpoint_name):
    checkpoint = torch.load(checkpoint_name)
    print("checkpoint exists!")
    print(checkpoint["test_loss"], checkpoint["best_test_loss"])
    exit()

best_test_loss = 10000
best_state_dict = None
for epoch in tqdm(range(args.epochs)):
    train_loss = train(grad_to_img_net, train_loader, sign, prune_rate=prune_rate, leak_batch=leak_batch)
    test_loss, reconstructed_imgs = test(grad_to_img_net, test_loader, sign, prune_rate=prune_rate, leak_batch=leak_batch)
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state_dict = deepcopy(grad_to_img_net.to("cpu").state_dict())
    grad_to_img_net.to("cuda")
    logger.info(f"epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}, best_test_loss: {best_test_loss}")
    if (epoch+1)%100 == 0:
        checkpoint = {}
        checkpoint["train_loss"] = train_loss
        checkpoint["test_loss"] = test_loss
        checkpoint["reconstructed_imgs"] = reconstructed_imgs
        checkpoint["gt_data"] = gt_data
        if args.trainset == "full":
            torch.save(checkpoint, f"checkpoint/{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}_best.pt")
        else:
            torch.save(checkpoint, f"checkpoint/{args.dataset}_{args.trainset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.pt")
    if (epoch+1) == int(0.75 * args.epochs):
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
checkpoint = {}
checkpoint["train_loss"] = train_loss
checkpoint["test_loss"] = test_loss
checkpoint["state_dict"] = grad_to_img_net.state_dict()
checkpoint["best_test_loss"] = best_test_loss
checkpoint["best_state_dict"] = best_state_dict
checkpoint["reconstructed_imgs"] = reconstructed_imgs
checkpoint["gt_data"] = gt_data
if args.trainset == "full":
    torch.save(checkpoint, f"checkpoint/{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.pt")
else:
    torch.save(checkpoint, f"checkpoint/{args.dataset}_{args.trainset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}.pt")
