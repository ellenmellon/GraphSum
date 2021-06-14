import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import GraphAttentionLayer
from transformers import *


class GAT(nn.Module):
    def __init__(self, ntypes, ntypes2, nhid, nclass, n_r_class, dropout, alpha, nheads, nlayers, n_r_types, finetune_scibert, ignore_title_node):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.ignore_title_node = ignore_title_node
        
        # node embeddings
        self.scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        if not finetune_scibert:
            for param in self.scibert.parameters():
                param.requires_grad = False
        self.token_l = nn.Linear(self.scibert.embeddings.word_embeddings.weight.shape[1], nhid * n_r_types)
        
        # feature: # mentions
        self.feature_emb = nn.Linear(1, nhid * n_r_types * 5) # this is just hard-coded to make the initial embedding with large enough dimention 
                                                                # and then converted into the same dimention as the hidden dimention (which is usually small)
        self.feature_l = nn.Linear(nhid * n_r_types * 5, nhid * n_r_types)

        self.type_emb = nn.Linear(ntypes, nhid * n_r_types * 5)
        self.type_l = nn.Linear(nhid * n_r_types * 5, nhid * n_r_types)

        self.type2_emb = nn.Linear(ntypes2, nhid * n_r_types * 5)
        self.type2_l = nn.Linear(nhid * n_r_types * 5, nhid * n_r_types)

        nn.init.xavier_normal_(self.type_emb.weight)
        nn.init.xavier_normal_(self.type2_emb.weight)
        nn.init.xavier_normal_(self.feature_emb.weight)


        # attention
        assert nheads == n_r_types
        self.attentions = [GraphAttentionLayer(nhid * n_r_types, nhid * n_r_types, dropout=dropout, alpha=alpha, n_heads=nheads) for _ in range(nlayers)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


        # out layers
        self.out = nn.Linear(nhid * n_r_types, nclass)

        self.edge_l = nn.Linear(2*nhid * n_r_types, nhid * n_r_types * 5)
        self.adj_l = nn.Linear(7, nhid * n_r_types * 5)
        self.edge_out = nn.Linear(nhid * n_r_types * 5, n_r_class)


    def forward(self, features, adj, tokens=None, types=None, types2=None):
        # embed nodes
        features = Variable(features.cuda())
        adj = Variable(adj.cuda())
        if tokens is not None:
            tokens = Variable(tokens.cuda())
        if types is not None:
            types = Variable(types.cuda())
            types2 = Variable(types2.cuda())


        features = self.feature_l(self.feature_emb(features))
        x = features
        if tokens is not None:
            if self.ignore_title_node:
                tokens = self.scibert(tokens)[1]
            else:
                token_type_ids = torch.zeros(tokens.shape, dtype=torch.long)
                token_type_ids[0, :] = 1
                token_type_ids = Variable(token_type_ids.cuda())
                # if there is title node, the first one should be title node and the type should be 1 for running scibert
                tokens = self.scibert(tokens, token_type_ids=token_type_ids)[1]
            tokens = self.token_l(tokens)
            x += tokens
        if types is not None:
            types = self.type_l(self.type_emb(types))
            x += types
            types2 = self.type2_l(self.type2_emb(types2))
            x += types2

        # attention layers
        x = F.dropout(x, self.dropout, training=self.training)
        for attn in self.attentions:
            x = attn(x, adj)
            x = x.squeeze(1)


        # classification layers
        
        x = F.dropout(x, self.dropout, training=self.training)
        node_score = self.out(x)
        node_score = F.log_softmax(node_score, dim=1)

        # TODO: currently (the following lines that calculate) edge_score are not used in training loss
        N = x.size()[0]
        encoded_edge = torch.cat([x.repeat(1, N).view(N * N, -1), x.repeat(N, 1)], dim=1).view(N * N, -1)
        encoded_adj = F.dropout(adj, self.dropout, training=self.training)
        encoded_adj = self.adj_l(encoded_adj.view(7, N*N).transpose(0,1))
        edge_score = self.edge_out(self.edge_l(encoded_edge)+encoded_adj)
        edge_score = F.log_softmax(edge_score, dim=1)
        
        return node_score, edge_score

