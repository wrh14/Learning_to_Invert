import argparse
import numpy as np
from tqdm import tqdm
import math
from copy import deepcopy
import os 
os.environ['KMP_WARNINGS'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from scipy.optimize import linear_sum_assignment

from utils import label_to_onehot, cross_entropy_for_onehot, CausalLoss, View, MEMLP, FasterMEMLP
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
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for training set-up')
parser.add_argument('--leak_mode', type=str, default="sign",
                    help='sign/prune-{prune_rate}/batch-{batch_size}')
parser.add_argument('--get_validation_rec_first_thou', type=str, default=None,
                    help='checkpoint path')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint path')
args = parser.parse_args()

if args.resume is not None:
    save_file_name = f"{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}_{args.seed}_{args.resume}"
else:
    save_file_name = f"{args.dataset}_{args.shared_model}_{args.model}_{args.leak_mode}_{args.lr}_{args.epochs}_{args.batch_size}_{args.seed}"

logger = set_logger("", filepath=f"print_logs/{save_file_name}.txt")
logger.info(args)
logger.info(f"logs are saved at print_logs/{save_file_name}.txt")

save_dir = "./"

def train(grad_to_img_net, net, data_loader, sign=False, mask=None, prune_rate=None, leak_batch=1):
    grad_to_img_net.train()
    total_loss = 0
    total_num = 0
#     total_correct_num = 0
    total_acc = 0
    if args.dataset.startswith("wikitext") or args.dataset.startswith("cola"):
        cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
    else:
        cross_entropy = torch.nn.CrossEntropyLoss()
    bin_stat = None
    for i, data in enumerate(tqdm(data_loader)):
        if args.dataset.startswith("wikitext") or args.dataset.startswith("cola"):
            images, labels = data["input_ids"], data["labels"]
        else:
            (images, labels) = data
        if pseudo:
            images = word_sampler.sample(images.shape)
            if args.dataset.startswith("wikitext"):
                labels = deepcopy(images)
            else:
                labels = (torch.rand([len(images)]) < 0.7044).long()
        xs, ys = leakage_dataset(images, labels, net)
        if start_i is not None:
            xs = torch.cat([xs[:, :start_i], xs[:, end_i:start_j]], dim=1)
        optimizer.zero_grad()
        batch_num = len(ys)
        batch_size = int(batch_num / leak_batch)
        batch_num = batch_size * leak_batch
        total_num += batch_num
        xs, ys = xs[:batch_num], ys[:batch_num]
        
        if sign:
            xs = torch.sign(xs)
        if prune_rate is not None:
            rank = torch.topk(xs.abs(), int(xs.size()[1] * (1 - prune_rate)), dim=1).indices
            mask = torch.zeros(xs.size())
            mask[torch.arange(len(ys)).view(-1, 1).expand(rank.size()), rank] = 1  
        if mask is not None:
            xs.mul_(mask)
        if selected_para is not None:
            xs = xs[:, selected_para]
            if gauss_noise > 0:
                xs = xs + torch.randn(*xs.shape) * gauss_noise
        else:
            if gauss_noise > 0:
                if bin_stat is None:
                    bin_stat = torch.sparse.mm(hashed_matrix.t(), torch.ones([1, xs.shape[1]]).t()).t().contiguous().squeeze()
            xs = torch.sparse.mm(hashed_matrix.t(), xs.t()).t().contiguous()
            if gauss_noise > 0:
                xs = xs + torch.randn(*xs.shape) * gauss_noise * torch.sqrt(bin_stat)
            
        xs = xs.view(batch_size, leak_batch, -1).mean(1).cuda()
        
        if args.dataset.startswith("wikitext") or args.dataset.startswith("cola"):
            preds = grad_to_img_net(xs).view(batch_size, leak_batch, num_tokens, -1)
            ys = ys.long().view(batch_size, leak_batch, num_tokens, 1).cuda()
            
            preds_repeat = preds.repeat(1, leak_batch, 1, 1).view(batch_size * leak_batch * leak_batch * num_tokens, -1)
            ys_repeat = ys.repeat(1, 1, leak_batch, 1).view(-1)
            per_data_loss = cross_entropy(preds_repeat, ys_repeat)
            if "bl" in args.dataset:
                per_data_loss /= word_probs[ys_repeat].cuda()
            
            batch_wise_ce = per_data_loss.view(batch_size, leak_batch, leak_batch, num_tokens).mean(-1)
            loss = 0
#             correct_num = 0
            acc = 0
            for batch_id, ce_mat in enumerate(batch_wise_ce):
                row_ind, col_ind = linear_sum_assignment(ce_mat.detach().cpu().numpy())
                loss += ce_mat[row_ind, col_ind].mean()
#                 correct_num += (preds[batch_id][col_ind].argmax(-1) == ys[batch_id][row_ind].squeeze(-1)).float().sum().item()
                
                per_data_correct = (preds[batch_id][col_ind].argmax(-1) == ys[batch_id][row_ind].squeeze(-1)).float()
                num_nonpad = len(ys[batch_id][row_ind].view(-1))
                acc = acc + per_data_correct.sum().item() / num_nonpad
                
            acc /= batch_size
            loss /= batch_size
            total_acc += acc
            if i % 10 == 0:
                logger.info(f"train iter: {i}; loss: {loss}; acc: {acc}")
        else:
            ys = ys.view(batch_size, leak_batch, -1).cuda()
            preds = grad_to_img_net(xs).view(batch_size, leak_batch, -1)
            batch_wise_mse = (torch.cdist(ys, preds) ** 2) / image_size
            loss = 0
            for mse_mat in batch_wise_mse:
                row_ind, col_ind = linear_sum_assignment(mse_mat.detach().cpu().numpy())
                loss += mse_mat[row_ind, col_ind].mean()
            loss /= batch_size
            if i % 10 == 0:
                logger.info(f"train iter: {i}; loss: {loss}")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_num
            
    total_loss = total_loss / total_num
    if args.dataset.startswith("wikitext") or args.dataset.startswith("cola"):
#         total_acc = total_correct_num / (total_num * num_tokens)
        total_acc = total_acc / len(data_loader)
        return (total_loss, total_acc)
    else:
        return (total_loss, None)


def test(grad_to_img_net, net, data_loader, sign=False, mask=None, prune_rate=None, leak_batch=1, num_test=None):
    grad_to_img_net.eval()
    total_loss = 0
    total_num = 0
    reconstructed_data = []
    gt_data = []
    acc_list = []
    #only for text data
    total_acc = 0
    if args.dataset.startswith("wikitext") or args.dataset.startswith("cola"):
        cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
    else:
        cross_entropy = torch.nn.CrossEntropyLoss()
    bin_stat = None
    if num_test is None:
        num_test = len(data_loader.dataset)
    for i, data in enumerate(tqdm(data_loader)):
        if i * args.batch_size >= num_test:
            break
        if args.dataset.startswith("wikitext") or args.dataset.startswith("cola"):
            images, labels = data["input_ids"], data["labels"]
        else:
            (images, labels) = data
        xs, ys = leakage_dataset(images, labels, net)
        if start_i is not None:
            xs = torch.cat([xs[:, :start_i], xs[:, end_i:start_j]], dim=1)
        with torch.no_grad():
            batch_num = len(ys)
            batch_size = int(batch_num / leak_batch)
            batch_num = batch_size * leak_batch
            total_num += batch_num
            xs, ys = xs[:batch_num], ys[:batch_num]
            if sign:
                xs = torch.sign(xs)
            if prune_rate is not None:
                mask = torch.zeros(xs.size())
#                 rank = torch.argsort(xs, dim=1)[:,  -int(xs.size()[1] * (1 - prune_rate)):]
                rank = torch.topk(xs.abs(), int(xs.size()[1] * (1 - prune_rate)), dim=1).indices
                mask[torch.arange(len(ys)).view(-1, 1).expand(rank.size()), rank] = 1   
            if mask is not None:
                xs = xs * mask
                
            if selected_para is not None:
                xs = xs[:, selected_para]
                if gauss_noise > 0:
                    xs = xs + torch.randn(*xs.shape) * gauss_noise
            else:
                if gauss_noise > 0:
                    if bin_stat is None:
                        bin_stat = torch.sparse.mm(hashed_matrix.t(), torch.ones([1, xs.shape[1]]).t()).t().contiguous().squeeze()
                xs = torch.sparse.mm(hashed_matrix.t(), xs.t()).t().contiguous()
                if gauss_noise > 0:
                    xs = xs + torch.randn(*xs.shape) * gauss_noise * torch.sqrt(bin_stat)
                
            if "norm" in args.dataset:
                xs = (xs - mean.cpu()) / (std.cpu() + 1e-5)
                
            xs = xs.view(batch_size, leak_batch, -1).mean(1).cuda()
            if args.dataset.startswith("wikitext") or args.dataset.startswith("cola"):
                preds = grad_to_img_net(xs).view(batch_size, leak_batch, num_tokens, -1)
                ys = ys.long().view(batch_size, leak_batch, num_tokens, 1).cuda()

                preds_repeat = preds.repeat(1, leak_batch, 1, 1).view(batch_size * leak_batch * leak_batch * num_tokens, -1)
                ys_repeat = ys.repeat(1, 1, leak_batch, 1).view(-1)
                per_data_loss = cross_entropy(preds_repeat, ys_repeat)

                batch_wise_ce = per_data_loss.view(batch_size, leak_batch, leak_batch, num_tokens).mean(-1)
                loss = 0
                acc = 0
                for batch_id, ce_mat in enumerate(batch_wise_ce):
                    row_ind, col_ind = linear_sum_assignment(ce_mat.detach().cpu().numpy())
                    loss += ce_mat[row_ind, col_ind].mean()
                    #save the reconstructed data in order
                    
                    per_data_correct = (preds[batch_id][col_ind].argmax(-1) == ys[batch_id][row_ind].squeeze(-1)).float()
                    num_nonpad = len(ys[batch_id][row_ind].view(-1))
#                     correct_num += per_data_correct.sum().item()
                    acc = acc + per_data_correct.sum().item() / num_nonpad
                    sorted_preds = preds[batch_id, col_ind].detach()
                    sorted_preds[row_ind] = preds[batch_id, col_ind]
                    reconstructed_data.append(sorted_preds.argmax(-1).cpu())
                loss /= batch_size
#                 acc = correct_num / (batch_size * leak_batch * num_tokens)
                acc = acc / batch_size
#                 total_correct_num += correct_num
                total_acc += acc
                if i % 10 == 0:
                    logger.info(f"test iter: {i}; loss: {loss}; acc: {acc}")
            else:
                ys = ys.view(batch_size, leak_batch, -1).cuda()
                grad_to_img_net.cuda()
                preds = grad_to_img_net(xs).view(batch_size, leak_batch, -1)
                batch_wise_mse = (torch.cdist(ys, preds) ** 2) / image_size
                loss = 0
                for batch_id, mse_mat in enumerate(batch_wise_mse):
                    row_ind, col_ind = linear_sum_assignment(mse_mat.detach().cpu().numpy())
                    loss += mse_mat[row_ind, col_ind].mean()
                    sorted_preds = preds[batch_id, col_ind]
                    sorted_preds[row_ind] = preds[batch_id, col_ind]
                    sorted_preds = sorted_preds.view(leak_batch, -1).detach().cpu()
                    reconstructed_data.append(sorted_preds)
                    gt_data.append(ys[batch_id])
                loss /= batch_size
#                 if i % 10 == 0:
#                     logger.info(f"test iter: {i}; loss: {loss}")
            total_loss += loss.item() * batch_num
            
    reconstructed_data = torch.cat(reconstructed_data)
    if args.dataset in ["MNIST", "FashionMNIST"]:
        reconstructed_data = reconstructed_data.view(-1, 1, 28, 28)
    elif args.dataset.startswith("CIFAR10"):
        reconstructed_data = reconstructed_data.view(-1, 3, 32, 32)

    total_loss = total_loss / total_num
    if args.dataset.startswith("wikitext") or args.dataset.startswith("cola"):
#         total_acc = total_correct_num / (total_num * num_tokens)
        total_acc = total_acc / len(data_loader)
        return (total_loss, total_acc), reconstructed_data
    else:
        return (total_loss, None), (reconstructed_data, gt_data)


#input the model shared among parties
if args.dataset in ["FashionMNIST", "MNIST"]:
    image_size = 784
    num_classes = 10
elif args.dataset.startswith("CIFAR10"):
    image_size = 3 * 32 * 32
    num_classes = 10
elif args.dataset.startswith("wikitext"):
    voc_size = 50257
    num_tokens = 16
elif args.dataset.startswith("cola"):
    voc_size = 30522
    num_tokens = 16
    num_classes = 2
    
#leakage mode
prune_rate = None
leak_batch = 1
sign = False
gauss_noise = 0
leak_mode_list = args.leak_mode.split("-")
single_infer = False
for i in range(len(leak_mode_list)):
    if leak_mode_list[i] == "sign":
        sign = True
    elif leak_mode_list[i] == "prune":
        prune_rate = float(leak_mode_list[i+1])
    elif leak_mode_list[i] == "batch":
        leak_batch = int(leak_mode_list[i+1])
    elif leak_mode_list[i] == "gauss":
        gauss_noise = float(leak_mode_list[i+1])
    elif leak_mode_list[i] == "singleinfer":
        single_infer = True
    
if args.shared_model == "ResNet20":
    net = resnet20(num_classes).to("cuda")
    compress_rate = 0.5
    torch.manual_seed(1234)
    net.apply(weights_init)
elif args.shared_model == "Transformer":
    compress_rate = 0.1
    model_checkpoint_name = "fl_settings/wikitext_shared_model.pt"
    net = torch.load(model_checkpoint_name)
elif args.shared_model == "BERT":
    compress_rate = 0.01
    model_checkpoint_name = "fl_settings/cola_shared_model.pt"
    net = torch.load(model_checkpoint_name)

model_size = 0
for i, parameters in enumerate(net.parameters()):
    if parameters.requires_grad:
        model_size += np.prod(parameters.size())
logger.info(f"model size: {model_size}")

logger.info("loading dataset...")
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
elif args.dataset.startswith("CIFAR10"):
    transform = transforms.Compose([
                transforms.ToTensor(),
                ])
    dst_train = datasets.CIFAR10("~/.torch", download=True, train=True, transform=transform)
    dst_validation = datasets.CIFAR10("~/.torch", download=True, train=False, transform=transform)    
    pseudo = False
elif args.dataset.startswith("wikitext"):
    from breaching.cases.data.datasets_text import _build_and_split_dataset_text
    config_train = torch.load(f"fl_settings/wikitext_train_dataset_config.pt")
    dst_train, _ = _build_and_split_dataset_text(config_train, "train", user_idx=0, return_full_dataset=True)
#     dst_train = torch.load(f"fl_settings/wikitext_train_dataset_new.pt")
    if len(args.dataset.split("-")) > 1:
        train_ratio = float(args.dataset.split("-")[1])
        torch.manual_seed(999)
        train_ids = torch.randperm(len(dst_train))[:int(len(dst_train) * train_ratio)]
        dst_train = dst_train.select(train_ids)
        if "pseudo" in args.dataset.split("-"):
            pseudo = True
        else:
            pseudo = False
    config_validation = torch.load(f"fl_settings/wikitext_validation_dataset_config.pt")
    dst_validation, _ = _build_and_split_dataset_text(config_validation, "validation", user_idx=0, return_full_dataset=True)
    
#     dst_test = torch.load(f"fl_settings/wikitext_test_dataset.pt")
#     dst_validation = torch.load(f"fl_settings/wikitext_validation_dataset_new.pt")
elif args.dataset.startswith("cola"):
    from breaching.cases.data.datasets_text import _build_and_split_dataset_text
    config_train = torch.load(f"fl_settings/cola_train_dataset_config.pt")
    dst_train, _ = _build_and_split_dataset_text(config_train, "train", user_idx=0, return_full_dataset=True)
    config_validation = torch.load(f"fl_settings/cola_validation_dataset_config.pt")
    dst_validation, _ = _build_and_split_dataset_text(config_validation, "validation", user_idx=0, return_full_dataset=True)
#     dst_train = torch.load(f"fl_settings/cola_train_dataset_new.pt")
#     dst_validation = torch.load(f"fl_settings/cola_validation_dataset_new.pt")
    if "pseudo" in args.dataset.split("-"):
        pseudo = True
    else:
        pseudo = False

batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(dataset=dst_train,
                                              batch_size=(batch_size * leak_batch), 
                                              shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=dst_validation,
                                              batch_size=(batch_size * leak_batch), 
                                              shuffle=False)

# function to compute the gradient
def leakage_dataset(images, labels, net):
    if not args.dataset.startswith("wikitext"):
        net.eval()
        criterion = cross_entropy_for_onehot
        batch_size = len(images)
        targets = torch.zeros([batch_size, images.view(batch_size, -1).size()[-1]])
        features = None

        for i, (image, label) in enumerate(zip(images, labels)):
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
            onehot_label = label_to_onehot(label, num_classes=num_classes)
            image, onehot_label = image.cuda(), onehot_label.cuda()
            pred = net(image)
            loss = criterion(pred, onehot_label)
            dy_dx = torch.autograd.grad(loss, [para for para in net.parameters() if para.requires_grad])
            original_dy_dx = torch.cat(list((_.detach().clone().view(-1) for _ in dy_dx)))
            targets[i] = image.view(-1)
            if features is None:
                features = torch.zeros([batch_size, len(original_dy_dx)])
            features[i] = original_dy_dx
    else:
        loss_fn = CausalLoss()
        batch_size = len(images)
        targets = torch.zeros([batch_size, images.view(batch_size, -1).size()[-1]])
        features = None
        for i, (image, label) in enumerate(zip(images, labels)):
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
            net.zero_grad()
            outputs = net(image)
            loss = loss_fn(outputs, label)
            loss.backward()
            grad = torch.cat([para.grad.view(-1) for para in net.parameters()])
            if features is None:
                features = torch.zeros([batch_size, len(grad)])
            features[i] = grad
            targets[i] = image
    return features, targets


if args.dataset.startswith("wikitext"):
    net.decoder.weight.data = net.encoder.weight.data
    start_i = 0
    end_i = 0
    start_j = 0
    end_j = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            if name == "encoder.weight":
                end_i = start_i + len(param.view(-1))
                break
            start_i += len(param.view(-1))
    for name, param in net.named_parameters():
        if param.requires_grad:
            if name == "decoder.weight":
                end_j = start_j + len(param.view(-1))
                break
            start_j += len(param.view(-1))
    model_size = start_i + start_j - end_i
else:
    start_i = None
    

#init the model
torch.manual_seed(args.seed)
if "hash" in args.dataset:
    hash_dim = int(model_size * compress_rate)
    shape = (model_size, hash_dim)
    hash_bin = torch.randint(0, hash_dim, (model_size,))
    i = torch.cat([torch.arange(model_size).long().unsqueeze(0), hash_bin.unsqueeze(0)], dim=0)
    hashed_matrix = torch.sparse_coo_tensor(i, torch.ones(model_size), shape)
    selected_para = None
else:
    selected_para = torch.randperm(model_size)[:int(model_size * compress_rate)]

if not args.dataset.startswith("CIFAR10"):
    if args.dataset.startswith("wikitext"):
        word_hist = torch.load("fl_settings/wiki_hist_word.pt")
    elif args.dataset.startswith("cola"):
        word_hist = torch.load("fl_settings/cola_hist_word.pt")
    word_probs = word_hist / word_hist.sum()
    if "pseudo" in args.dataset:
        word_sampler = torch.distributions.categorical.Categorical(word_probs)

if args.model.startswith("MLP"):
    hidden_size = int(args.model.split("-")[-1])
    output_size = image_size if single_infer else image_size * leak_batch
    grad_to_img_net = nn.Sequential(
        nn.Linear(int(model_size * compress_rate), hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )
    grad_to_img_net = grad_to_img_net.cuda()
elif args.model.startswith("NLPMLP"):
    hidden_size = int(args.model.split("-")[1])
    embed_size = int(args.model.split("-")[2])
    output_size = num_tokens * voc_size
    grad_to_img_net = nn.Sequential(
        nn.Linear(int(model_size * compress_rate), hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, leak_batch * num_tokens * embed_size),
        nn.ReLU(),
        View([leak_batch, num_tokens, embed_size]),
        nn.Linear(embed_size, voc_size)
    )
    grad_to_img_net = grad_to_img_net.cuda()
    
logger.info(f"input size: {int(model_size * compress_rate)}; output size: {output_size}")


size = 0
for parameters in grad_to_img_net.parameters():
    size += np.prod(parameters.size())
logger.info(f"net size: {size}")

#training set-up
lr = args.lr
epochs = args.epochs
optimizer = torch.optim.Adam(grad_to_img_net.parameters(), lr=lr)
    
if args.get_validation_rec_first_thou is not None:
    validation_loader = torch.utils.data.DataLoader(dataset=dst_validation,
                                              batch_size=(batch_size * leak_batch), 
                                              shuffle=False)
    checkpoint = torch.load(args.get_validation_rec_first_thou)
    grad_to_img_net.load_state_dict(checkpoint["state_dict"])
    (val_loss, val_acc), reconstructed_imgs = test(grad_to_img_net, net, validation_loader, sign, prune_rate=prune_rate, leak_batch=leak_batch, num_test=4000)
    checkpoint["val_loss"] = val_loss
    checkpoint["val_acc"] = val_acc
    checkpoint["val_reconstructed_imgs"] = reconstructed_imgs
    checkpoint["val_gt_data"] = dst_validation["labels"]
    torch.save(checkpoint, f"{args.get_validation_rec_first_thou}_val")
    exit()
    
if args.resume is not None:
    checkpoint = torch.load(f"checkpoint/{args.resume}")
    grad_to_img_net.load_state_dict(checkpoint["state_dict"])
    
best_test_loss = 1000000
best_state_dict = None
for epoch in tqdm(range(args.epochs)):
    train_loss, train_acc = train(grad_to_img_net, net, train_loader, sign, prune_rate=prune_rate, leak_batch=leak_batch)
    (test_loss, test_acc), reconstructed_imgs = test(grad_to_img_net, net, test_loader, sign, prune_rate=prune_rate, leak_batch=leak_batch)
    grad_to_img_net = grad_to_img_net.cpu()
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state_dict = deepcopy(grad_to_img_net).cpu().state_dict()
    logger.info(f"epoch: {epoch}, train_loss: {train_loss}, train_acc: {train_acc}, test_loss: {test_loss}, test_acc: {test_acc}, best_test_loss: {best_test_loss}")
    if (epoch+1) == int(0.5 * args.epochs):
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
#     checkpoint = {}
#     checkpoint["train_loss"] = train_loss
#     checkpoint["val_loss"] = test_loss
#     checkpoint["train_acc"] = train_acc
#     checkpoint["val_acc"] = test_acc
#     checkpoint["state_dict"] = grad_to_img_net.state_dict()
#     checkpoint["best_test_loss"] = best_test_loss
#     checkpoint["best_state_dict"] = best_state_dict
#     checkpoint["optimizer_state_dict"] = optimizer.state_dict()
#     if args.dataset.startswith("wikitext") or args.dataset.startswith("cola"):
#         checkpoint["val_reconstructed_imgs"] = reconstructed_imgs
#         checkpoint["gt_data"] = dst_validation["labels"]
#     elif args.dataset.startswith("CIFAR10"):
#         checkpoint["reconstructed_imgs"] = reconstructed_imgs[0]
#         checkpoint["gt_data"] = reconstructed_imgs[1]
#     checkpoint["epoch"] = epoch
#     torch.save(checkpoint, f"{save_dir}/checkpoint/{save_file_name}_version1.pt")
#     torch.save(checkpoint, f"{save_dir}/checkpoint/{save_file_name}_version2.pt")
    grad_to_img_net = grad_to_img_net.cuda()
#     del checkpoint

