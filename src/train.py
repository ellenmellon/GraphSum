from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import json
from utils import get_all_feature_fnames, load_data
from eval import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from transformers import *

from models import GAT
from env import DATA_DIR, GRAPH_DIR


# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=0.00005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--n_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--n_layers', type=int, default=6, help='Number of attention layers.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--ignore_title_node', action='store_true', default=True, help='Disables adding paper title node.')
parser.add_argument('--ignore_node_type', action='store_true', default=False, help='Disables node type in node embedding.')
parser.add_argument('--ignore_node_name', action='store_true', default=False, help='Disables node string name in node embedding.')
parser.add_argument('--finetune_scibert', action='store_true', default=False, help='Finetuning scibert when training.')
parser.add_argument('--model_output_dir', default='output', type=str, help="Directory for saving models.")
parser.add_argument('--neg_sampling_ratio', type=float, default=3.0, help='Negative sampling ratio.')
parser.add_argument('--neg_sampling_ratio_edge', type=float, default=2.0, help='Negative sampling ratio for edge learning.')
parser.add_argument('--beta', type=float, default=0.0, help='Weight for edge prediction loss. Default is ignoring edge prediction loss.')

# little need to change these 
parser.add_argument('--batch_size', type=int, default=10, help='Number of steps fpr every backpropogate.')
parser.add_argument('--steps', type=int, default=100000, help='Number of steps to train.')
parser.add_argument('--eval_every_n_steps', type=int, default=50, help='Evaluate after n steps.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--max_n_nodes', type=int, default=400, help='Skips examples with too many nodes')
parser.add_argument('--max_n_nodes_test', type=int, default=500, help='Truncate test examples with too many nodes')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--n_valid_graphs', type=int, default=200, help='Number of graphs to be in valid set.')
parser.add_argument('--n_train_graphs_to_load_per_time', type=int, default=100, help='Number of training graphs to be loaded each time. (depending on cpu memory)')
parser.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm.")




args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not args.ignore_title_node:
    args.max_n_nodes += 1

# Load data

# TODO: replace such hard-coded numbers ...
n_types  = 6 # 6 entity types
n_types2 = 20 # up to 20 section ids
n_r_types = 7 # 7 relation types
if not args.ignore_title_node:
    n_types += 1
    n_r_types += 1
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# Model and optimizer
model = GAT(ntypes=n_types,
            ntypes2=n_types2,
            nhid=args.hidden, 
            nclass=2,      # hard-coded for # classes (salient or not salient)
            n_r_class=n_r_types+2,    # hard-coded for # relation classes (7 classes + 'no relation' + 'coref')
            dropout=args.dropout,
            alpha=args.alpha,
            nheads=n_r_types,  # to map each head -> relation type
            nlayers=args.n_layers,
            n_r_types=n_r_types,
            finetune_scibert=args.finetune_scibert,
            ignore_title_node=args.ignore_title_node)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()



random.seed(1234)
train_data_path = '{}/json/train/processed/graph-sum'.format(DATA_DIR)
valid_data_path = '{}/json/valid/processed/graph-sum'.format(DATA_DIR)
train_fnames = get_all_feature_fnames(train_data_path)
valid_fnames = get_all_feature_fnames(valid_data_path)


random.shuffle(train_fnames)
n_train_graphs = args.n_train_graphs_to_load_per_time
train_fnames_list = [train_fnames[i*n_train_graphs:(i+1)*n_train_graphs] for i in range(int(len(train_fnames)/n_train_graphs)+1)]

valid_adj_list, valid_features_list, valid_labels_list, valid_r_labels_list, valid_token_ids_list, valid_types_list, valid_types2_list, included_valid_fnames = load_data(valid_fnames, 
                                                                                                           tokenizer=tokenizer, 
                                                                                                           n_types=n_types, 
                                                                                                           n_r_types=n_r_types,
                                                                                                           max_n_nodes=args.max_n_nodes,
                                                                                                           ignore_node_type=args.ignore_node_type, 
                                                                                                           ignore_node_name=args.ignore_node_name,
                                                                                                           ignore_title_node=args.ignore_title_node)
n_valid = len(valid_features_list)
valid_full_graph_fname = '{}/json/valid/processed/merged_entities/full_graph.json.ent_merged'.format(DATA_DIR)
docid2full = get_docid2full(valid_full_graph_fname)

gold_graph_fname = '{}/gold/valid'.format(GRAPH_DIR)
included_docids = [fname.split('/')[-1].replace('.features', '').strip() for fname in included_valid_fnames]
docid2goldgraph = get_docid2graph(gold_graph_fname)




class TrainDataloader():
    def __init__(self):
        self.idx = 1
        self.list_idx = 1
        self.repeat = 0
        train_data = load_data(train_fnames_list[0], 
                               tokenizer=tokenizer, 
                               n_types=n_types, 
                               n_r_types=n_r_types,
                               max_n_nodes=args.max_n_nodes,
                               ignore_node_type=args.ignore_node_type, 
                               ignore_node_name=args.ignore_node_name,
                               ignore_title_node=args.ignore_title_node)
        train_adj_list, train_features_list, train_labels_list, train_r_labels_list, train_token_ids_list, train_types_list, train_types2_list, _ = train_data
        self.train_adj_list = train_adj_list
        self.train_features_list = train_features_list
        self.train_labels_list = train_labels_list
        self.train_r_labels_list = train_r_labels_list
        self.train_token_ids_list = train_token_ids_list
        self.train_types_list = train_types_list
        self.train_types2_list = train_types2_list

        self.curr = (self.train_features_list[0], 
                     self.train_adj_list[0], 
                     self.train_labels_list[0], 
                     self.train_r_labels_list[0], 
                     self.train_token_ids_list[0], 
                     self.train_types_list[0],
                     self.train_types2_list[0])
    
    def get_next_train_data(self):
        if self.repeat < 1:
            self.repeat += 1
            return self.curr
        else:
            if self.idx >= len(self.train_adj_list):
                train_data = load_data(train_fnames_list[self.list_idx], 
                                       tokenizer=tokenizer, 
                                       n_types=n_types, 
                                       n_r_types=n_r_types,
                                       max_n_nodes=args.max_n_nodes,
                                       ignore_node_type=args.ignore_node_type, 
                                       ignore_node_name=args.ignore_node_name,
                                       ignore_title_node=args.ignore_title_node)
                train_adj_list, train_features_list, train_labels_list, train_r_labels_list, train_token_ids_list, train_types_list, train_types2_list, _ = train_data
                self.train_adj_list = train_adj_list
                self.train_features_list = train_features_list
                self.train_labels_list = train_labels_list
                self.train_r_labels_list = train_r_labels_list
                self.train_token_ids_list = train_token_ids_list
                self.train_types_list = train_types_list
                self.train_types2_list = train_types2_list
                self.list_idx = (self.list_idx+1)%len(train_fnames_list)
                self.idx = 0

            self.curr = (self.train_features_list[self.idx], 
                         self.train_adj_list[self.idx], 
                         self.train_labels_list[self.idx], 
                         self.train_r_labels_list[self.idx], 
                         self.train_token_ids_list[self.idx], 
                         self.train_types_list[self.idx],
                         self.train_types2_list[self.idx])
            self.idx += 1
            self.repeat = 1
            return self.curr



def train(step, train_loader, batch_size):
    #t = time.time()
    model.train()
    optimizer.zero_grad()
    for i in range(batch_size):
        train_features, train_adj, train_labels, train_r_labels, train_token_ids, train_types, train_types2 = train_loader.get_next_train_data()
        node_output, edge_output = model(train_features, 
                                         train_adj, 
                                         tokens=train_token_ids, 
                                         types=train_types,
                                         types2=train_types2)

        # calc node loss
        p_indices = (train_labels.flatten() == 1).nonzero().flatten()
        n_indices = (train_labels.flatten() == 0).nonzero().flatten()
        n_indices = n_indices[torch.randperm(n_indices.shape[0])][:int(len(p_indices)*args.neg_sampling_ratio)+1] # plus 1 to avoid empty cases
        indices = torch.cat((n_indices, p_indices))
        node_loss_train = F.nll_loss(node_output[indices], Variable(train_labels.cuda())[indices])
  
  
        # calc edge loss
        if args.beta > 0.0:
            try:
                p_r_indices = (train_r_labels.flatten() < n_r_types).nonzero().flatten()
                n_r_indices = (train_r_labels.flatten() >= n_r_types).nonzero().flatten().tolist()
                
                valid_n_indices = []
                for a in p_indices:
                    for b in p_indices:
                        valid_n_indices += [int(a)*node_output.shape[0]+int(b)]
                
                n_r_indices = torch.Tensor(list(set(valid_n_indices).intersection(set(n_r_indices)))).type_as(p_indices).to(p_indices.device)
                n_r_indices = n_r_indices[torch.randperm(n_r_indices.shape[0])][:int(len(p_r_indices)*args.neg_sampling_ratio_edge)+1]
          
                r_indices = torch.cat((n_r_indices, p_r_indices))
                edge_loss_train = F.nll_loss(edge_output[r_indices], Variable(train_r_labels.cuda())[r_indices])
                loss_train = node_loss_train + args.beta * edge_loss_train
            except:
                print('Empty edge loss!!!')
                loss_train = node_loss_train
        else:
            loss_train = node_loss_train
      
        loss_train.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()


    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    ave_f1_val = None
    ave_r_f1_val = None
    if (step+1)%args.eval_every_n_steps == 0:
        model.eval()
        ave_loss_val = 0.0
        ave_f1_val = 0.0
        ave_p_val = 0.0
        ave_r_val = 0.0
        ave_r_f1_val = 0.0
        ave_r_p_val = 0.0
        ave_r_r_val = 0.0
        
        for i in range(n_valid):
            node_output, edge_output = model(valid_features_list[i], 
                                             valid_adj_list[i], 
                                             tokens=valid_token_ids_list[i], 
                                             types=valid_types_list[i],
                                             types2=valid_types2_list[i])

            node_loss_val = F.nll_loss(node_output, Variable(valid_labels_list[i].cuda()))
            edge_loss_val = F.nll_loss(edge_output, Variable(valid_r_labels_list[i].cuda()))
            loss_val = node_loss_val + args.beta * edge_loss_val

            
            docid = included_docids[i]
            gold_graph = docid2goldgraph[docid]

            preds = (node_output.max(1)[1].flatten() == 1).nonzero().flatten().tolist()

            full_graph = docid2full[docid]
            entities = get_entities_map(full_graph, preds)
            if args.beta == 0.0:
                relations = find_valid_relations(full_graph, entities)
            else:
                relations = find_valid_relations_from_predicted_rels(edge_output, entities, n_r_types, node_output.shape[0])
            pred_graph = {'entities':entities, 'relations':relations}

            result = eval_graph_hard_align(gold_graph, pred_graph, docid, 'valid', match_type=True)

            ave_loss_val += loss_val.data.item()
            ave_p_val += result['entity']['precision']
            ave_r_val += result['entity']['recall']
            ave_f1_val += result['entity']['f1']
            ave_r_p_val += result['relation']['precision']
            ave_r_r_val += result['relation']['recall']
            ave_r_f1_val += result['relation']['f1']
            
            

        print('step: {:04d}'.format(step+1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'loss_node_train: {:.4f}'.format(node_loss_train.data.item()),
              'ave_loss_val: {:.4f}'.format(ave_loss_val / n_valid),
              'total time trained: {:.4f}s'.format(time.time() - t_total))
        print('    ',
              'ave_p_val : {:.4f}'.format(ave_p_val / n_valid),
              'ave_r_val : {:.4f}'.format(ave_r_val / n_valid),
              'ave_f1_val : {:.4f}'.format(ave_f1_val / n_valid),
              'ave_r_p_val : {:.4f}'.format(ave_r_p_val / n_valid),
              'ave_r_r_val : {:.4f}'.format(ave_r_r_val / n_valid),
              'ave_r_f1_val : {:.4f}'.format(ave_r_f1_val / n_valid)) 

    if ave_r_f1_val is None:
        return None
    else:
        return ave_r_f1_val/n_valid



""" Train model """

if os.path.exists(args.model_output_dir):
    raise Exception('Output directory {} exists !'.format(args.model_output_dir))

os.mkdir(args.model_output_dir)
with open('{}/args.txt'.format(args.model_output_dir), 'w') as fout:
    for arg in vars(args):
        fout.write('{}={}\n'.format(arg, getattr(args, arg)))

t_total = time.time()
f1_values = []
bad_counter = 0
best = 0.0
best_step = 0
train_loader = TrainDataloader()
for step in range(args.steps):
    f1 = train(step, train_loader, args.batch_size)
    if f1 is None:
        continue
    f1_values.append(f1)

    
    if f1_values[-1] > best:
        torch.save(model.state_dict(), '{}/{}.pkl'.format(args.model_output_dir, step))
        best = f1_values[-1]
        best_step = step
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('{}/*.pkl'.format(args.model_output_dir))
    for file in files:
        step_nb = int(file.split('/')[-1].split('.')[0])
        if step_nb < best_step:
            os.remove(file)

files = glob.glob('{}/*.pkl'.format(args.model_output_dir))
for file in files:
    step_nb = int(file.split('/')[-1].split('.')[0])
    if step_nb > best_step:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

