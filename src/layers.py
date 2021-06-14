import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module): # Multihead
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, n_heads, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = torch.tensor(in_features, requires_grad=False).float()
        self.out_features = out_features
        self.n_heads = n_heads
        self.alpha = alpha
        self.concat = concat


        self.query_layer = nn.Linear(in_features, out_features, bias=False)

        self.key_layer = nn.Linear(in_features, out_features, bias=False)

        self.value_layer = nn.Linear(in_features, out_features, bias=False)


    def forward(self, input, adj=None):
        N = input.size()[0]
        keys = input.repeat(N,1,1).view(N,N,-1).clone().detach().requires_grad_(False)
        input = input.unsqueeze(1)


        Q = self.query_layer(input)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self.out_features / self.n_heads)

        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))

        # normalize with sqrt(dk)
        # attention and _key_dim should be in the same device.
        attention = attention / torch.sqrt(self.in_features).to(input.get_device())

        if adj is not None:
            assert adj.shape[0] == self.n_heads
            adj = adj.view(attention.shape[0], -1).unsqueeze(1)
            zero_vec = -9e15*torch.ones_like(attention)
            #attention = (adj + zero_vec) * attention
            attention = torch.where(adj > 0, attention, zero_vec)
    
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(attention, self.dropout/(1.0*self.n_heads), training=self.training)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
    
    
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self.n_heads)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)


        attention = attention + input

        return F.elu(attention)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

