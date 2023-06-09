import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.nn.parameter import Parameter
# import torch.utils.data.Dataset as Dataset
from typing import Optional
import time

IMAGE_SIZE = {
    "MNIST": [1, 28, 28],
    "FashionMNIST": [1, 28, 28],
    "CIFAR10": [3, 32, 32],
}

class CausalLoss(torch.nn.Module):
    """Cross Entropy variant for next-token prediction in causal language modeling."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """If no labels are given, then the same sequence is re-used."""
        # Based on https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L1069
        # Shift so that tokens < n predict n
#         print("outputs", outputs.shape)
        shift_logits = outputs[:, :-1, :].contiguous()
        if labels is None:
            shift_labels = outputs[:, 1:].contiguous()
        elif labels.dtype == torch.long:
            shift_labels = labels[:, 1:].contiguous().view(-1)
        else:
            shift_labels = labels[:, 1:, :].contiguous().view(-1, labels.shape[-1])
        # Flatten the tokens
        return self.loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels)

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

class View(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

class MEMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, num_tokens, voc_size, compress_rate):
        super(MEMLP, self).__init__()
        num_part = int(input_size / hidden_size)
        self.num_part = num_part
        compress_hidden_size = int(compress_rate * hidden_size)
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.voc_size = voc_size
        self.compress_hidden_size = compress_hidden_size
        linear_layers = [nn.Linear(hidden_size, compress_hidden_size) for i in range(num_part - 1)]
#         linear_layers = [nn.Linear(num_part-1, hidden_size, compress_hidden_size)]
        linear_layers.append(nn.Linear(input_size - (num_part - 1) * hidden_size, compress_hidden_size))
        self.linear_layers = nn.ModuleList(linear_layers)
        self.linear_2 = nn.Linear(compress_hidden_size*num_part, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, num_tokens * embed_size)
        self.linear_4 = nn.Linear(embed_size, voc_size)
        self.relu = nn.ReLU()
        torch.manual_seed(520)
        permutation = torch.randperm(input_size)
        self.shuffle = permutation
        torch.manual_seed(int(time.time()))
        
    def forward(self, xs):
        B = xs.shape[0]
#         xs = xs[:, self.shuffle]
        hidden_xs = []
        for i in range(self.num_part - 1):
            hidden_xs.append(self.linear_layers[i](xs[:, i*self.hidden_size:(i+1)*self.hidden_size]))
#         hidden_xs = [self.linear_layers[0](xs[:, :(self.num_part-1)*self.hidden_size].reshape(-1, self.num_part - 1, self.hidden_size).permute(1, 0, 2)).permute(1, 0, 2).view(B, -1)]
        
        hidden_xs.append(self.linear_layers[-1](xs[:, (self.num_part-1) * self.hidden_size:]))
        hidden_xs = self.relu(torch.cat(hidden_xs, dim=1))
        hidden_xs = self.relu(self.linear_2(hidden_xs))
        hidden_xs = self.relu(self.linear_3(hidden_xs))
        hidden_xs = hidden_xs.view(B, -1, self.embed_size)
        logits = self.linear_4(hidden_xs)
        return logits

    
class BatchLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, num_linear, bias=True, seed=0):
        super(BatchLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_linear = num_linear
        self.weight = Parameter(torch.empty((num_linear, in_features, out_features) ))
        self.bias = None
#         self.bias = Parameter(torch.empty(num_linear, out_features))
        
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        init.uniform_(self.weight, -bound, bound)
#         if self.bias is not None:
#             init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
#         return (torch.bmm(input, self.weight).permute(1, 0, 2) + self.bias).permute(1, 0, 2)
        return torch.bmm(input, self.weight)

    def extra_repr(self):
        return 'in_features={}, out_features={}, num_linear={}, bias={}'.format(
            self.in_features, self.out_features, self.num_linear, self.bias is not None
        )


class FasterMEMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, num_tokens, voc_size, compress_rate):
        super(FasterMEMLP, self).__init__()
        num_part = int(input_size / hidden_size)
        self.num_part = num_part
        compress_hidden_size = int(compress_rate * hidden_size)
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.voc_size = voc_size
        self.compress_hidden_size = compress_hidden_size
#         linear_layers = [nn.Linear(hidden_size, compress_hidden_size) for i in range(num_part - 1)]
        linear_layers = [BatchLinear(hidden_size, compress_hidden_size, num_part-1)]
#         linear_layers = [nn.Linear(num_part-1, hidden_size, compress_hidden_size)]
        linear_layers.append(nn.Linear(input_size - (num_part - 1) * hidden_size, compress_hidden_size))
        self.linear_layers = nn.ModuleList(linear_layers)
        self.linear_2 = nn.Linear(compress_hidden_size*num_part, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, num_tokens * embed_size)
        self.linear_4 = nn.Linear(embed_size, voc_size)
        self.relu = nn.ReLU()
        torch.manual_seed(520)
        permutation = torch.randperm(input_size)
        self.shuffle = permutation
        torch.manual_seed(int(time.time()))
        
    def forward(self, xs):
        B = xs.shape[0]
#         xs = xs[:, self.shuffle]
#         hidden_xs = []
#         for i in range(self.num_part - 1):
#             hidden_xs.append(self.linear_layers[i](xs[:, i*self.hidden_size:(i+1)*self.hidden_size]))
        hidden_xs = [self.linear_layers[0](xs[:, :(self.num_part-1)*self.hidden_size].reshape(-1, self.num_part - 1, self.hidden_size).permute(1, 0, 2)).permute(1, 0, 2).contiguous().view(B, -1)]
#         print("1", hidden_xs[0])
#         hidden_xs = [self.linear_layers[0](xs[:, :(self.num_part-1)*self.hidden_size].reshape(-1, self.num_part - 1, self.hidden_size).permute(1, 0, 2)).permute(1, 0, 2).view(B, -1)]
        
        hidden_xs.append(self.linear_layers[-1](xs[:, (self.num_part-1) * self.hidden_size:]))
        hidden_xs = self.relu(torch.cat(hidden_xs, dim=1))
#         print("2", hidden_xs)
        hidden_xs = self.relu(self.linear_2(hidden_xs))
#         print("3", hidden_xs)
        hidden_xs = self.relu(self.linear_3(hidden_xs))
#         print("4", hidden_xs)
        hidden_xs = hidden_xs.view(B, -1, self.embed_size)
#         print("5", hidden_xs)
        logits = self.linear_4(hidden_xs)
#         print("6", logits)
        return logits
    
    
class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR thingies."""

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=[1, 2, 2, 2], pool='avg'):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    
# class WikitextGradToTokenDataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, data_dir, split="train"):
#         self.split = split
#         self.data_dir = data_dir
        
#     def __len__(self):
#         if self.split == "train":
#             return 1801350
#         elif self.split == "validation":
#             return 15045
#         else:
#             return 17100
        
#     def __getitem__(self, idx):
#         checkpoint = torch.load(f"{self.data_dir}/{split}/{idx}.pt")
#         return checkpoint["feature"], checkpoint["target"]
    