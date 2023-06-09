import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import numpy as np

class HashedExpand(torch.autograd.Function):
    def __init__(self, in_features, out_features, bins, indices_hash):
        super(HashedExpand, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bins = bins
        self.indices_hash = indices_hash

    def forward(self, x):
        return torch.index_select(x, 0, self.indices_hash).view(self.out_features, self.in_features)

    def backward(self, grad):
        grad = grad.view(self.in_features * self.out_features)
        true_grad = grad.new(self.bins).fill_(0)
        true_grad.put_(self.indices_hash, grad, accumulate=True)
        return true_grad


class HashedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bins, bias=True, seed=0, hash_type='hashed_bins', hash_rate=1.0):
        super(HashedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bins = bins
        self.weight = Parameter(torch.Tensor(bins))
        self.hash_type = hash_type
        self.hash_rate = hash_rate
        torch.manual_seed(seed)
        if self.hash_type == 'hashed_bins':
            #print("hashed_linear type = hashed_bins")
            self.indices_hash = torch.randint(0, bins, (int(out_features * in_features * hash_rate),)).long()
        elif self.hash_type == 'permutation':
            self.perm_num = int(out_features * in_features * hash_rate)
            self.perm_num = max(1, self.perm_num)
            if hash_rate == 1.0:
                #mege the original case
                self.perm_para = torch.arange(out_features * in_features).long()
            else:
                torch.manual_seed(seed + 888)
                self.perm_para = torch.randperm(out_features * in_features)[: (self.perm_num - 1)]
            torch.manual_seed(seed)
            #print("hashed_linear type = permutation")
            self.indices_hash = torch.arange(out_features * in_features).long()
            self.indices_hash[self.perm_para] = self.indices_hash[self.perm_para][torch.randperm(self.perm_num)]
        else:
            print('invalid hash type')
            exit()
        if torch.cuda.is_available():
            self.indices_hash = self.indices_hash.cuda()
        self.hash_scatter = HashedExpand(in_features, out_features, bins, self.indices_hash)
        setattr(self, 'indices_hash_seed_' + str(seed), self.indices_hash)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.out_features)
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        true_weight = self.hash_scatter(self.weight)
        return F.linear(input, true_weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def reset_seed(self, seed):
        if hasattr(self, 'indices_hash_seed_' + str(seed)):
            self.indices_hash = getattr(self, 'indices_hash_seed_' + str(seed))
            self.hash_scatter = HashedExpand(self.in_features, self.out_features, self.bins, self.indices_hash)
        else:
            torch.manual_seed(seed)

            if self.hash_type == 'hashed_bins':
                self.indices_hash = torch.randint(0, self.bins, (self.out_features * self.in_features,)).long().cuda()
            elif self.hash_type == 'permutation':
                self.indices_hash = torch.arange(self.out_features * self.in_features).long().cuda()
                self.indices_hash[self.perm_para] = self.indices_hash[self.perm_para][torch.randperm(self.perm_num)].cuda()
            else:
                print('invalid hash type')
                exit()
            self.hash_scatter = HashedExpand(self.in_features, self.out_features, self.bins, self.indices_hash)
            setattr(self, 'indices_hash_seed_' + str(seed), self.indices_hash)
