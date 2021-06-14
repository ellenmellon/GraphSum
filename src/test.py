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

parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--n_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--n_layers', type=int, default=6, help='Number of attention layers.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--ignore_title_node', action='store_true', default=True, help='Disables adding paper title node.')
parser.add_argument('--ignore_node_type', action='store_true', default=False, help='Disables node type in node embedding.')
parser.add_argument('--ignore_node_name', action='store_true', default=False, help='Disables node string name in node embedding.')
parser.add_argument('--finetune_scibert', action='store_true', default=False, help='Finetuning scibert when training.')
parser.add_argument('--model_output_dir', default='output', type=str, help="Directory for saving models.")
parser.add_argument('--beta', type=float, default=0.0, help='Weight for edge prediction loss.')

# little need to change these 
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--max_n_nodes', type=int, default=400, help='Skips examples with too many nodes')
parser.add_argument('--max_n_nodes_test', type=int, default=500, help='Truncate test examples with too many nodes')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')



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

if args.cuda:
    model.cuda()

def compute_test(fout, fout_pred, corpus_type, included_docids, docid2full):
    model.eval()

    ave_f1_test = 0.0
    ave_p_test = 0.0
    ave_r_test = 0.0
    ave_r_f1_test = 0.0
    ave_r_p_test = 0.0
    ave_r_r_test = 0.0
    
    gold_graph_fname = '{}/gold/{}'.format(GRAPH_DIR, corpus_type)
    
    docid2goldgraph = get_docid2graph(gold_graph_fname)
    
    
    #assert n_test == len(test_fnames) 
    print(n_test, len(test_fnames))
    for i in range(n_test):

        node_output, edge_output = model(test_features_list[i], 
                                         test_adj_list[i], 
                                         tokens=test_token_ids_list[i], 
                                         types=test_types_list[i],
                                         types2=test_types2_list[i])
        

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


        result = eval_graph_hard_align(gold_graph, pred_graph, docid, corpus_type, match_type=True)
        
        
   
        pred_dict =  {'doc_key':docid, 'eids':preds}
        if args.beta > 0.0:
            new_relations = {}
            for rel in relations:
                new_relations['{}_{}'.format(rel[0], rel[1])] = list(relations[rel])
            pred_dict['relations'] = new_relations
        fout_pred.write(json.dumps(pred_dict)+'\n')


        
        ave_p_test += result['entity']['precision']
        ave_r_test += result['entity']['recall']
        ave_f1_test += result['entity']['f1']
        
        ave_r_p_test += result['relation']['precision']
        ave_r_r_test += result['relation']['recall']
        ave_r_f1_test += result['relation']['f1']

    print("Test set results:",
          "ave_p= {:.4f}".format(ave_p_test / n_test),
          "ave_r= {:.4f}".format(ave_r_test / n_test),
          "ave_f1= {:.4f}".format(ave_f1_test / n_test),
          "ave_r_p= {:.4f}".format(ave_r_p_test / n_test),
          "ave_r_r= {:.4f}".format(ave_r_r_test / n_test),
          "ave_r_f1= {:.4f}".format(ave_r_f1_test / n_test))
   
    if fout is not None:
        fout.write('Test set results:\n')
        fout.write("ave_p= {:.4f}\n".format(ave_p_test / n_test))
        fout.write("ave_r= {:.4f}\n".format(ave_r_test / n_test))
        fout.write("ave_f1= {:.4f}\n".format(ave_f1_test / n_test))
        fout.write("ave_r_p= {:.4f}\n".format(ave_r_p_test / n_test))
        fout.write("ave_r_r= {:.4f}\n".format(ave_r_r_test / n_test))
        fout.write("ave_r_f1= {:.4f}\n".format(ave_r_f1_test / n_test))

if not os.path.exists(args.model_output_dir):
    raise Exception('Model directory {} does not exists !'.format(args.model_output_dir))


files = glob.glob('{}/*.pkl'.format(args.model_output_dir))
best_step = 0
for file in files:
    step_nb = int(file.split('/')[-1].split('.')[0])
    if step_nb > best_step:
        best_step = step_nb


""" Restore best model """
print('Loading {}th step'.format(best_step))
model.load_state_dict(torch.load('{}/{}.pkl'.format(args.model_output_dir, best_step)))



""" Testing """
for corpus_type in ['test', 'auto_test']:
    test_data_path = '{}/json/{}/processed/graph-sum/'.format(DATA_DIR, corpus_type)
    test_full_graph_fname = '{}/json/{}/processed/merged_entities/full_graph.json.ent_merged'.format(DATA_DIR, corpus_type)
    docid2full = get_docid2full(test_full_graph_fname)
    test_fnames = get_all_feature_fnames(test_data_path)
    test_adj_list, test_features_list, test_labels_list, test_r_labels_list, test_token_ids_list, test_types_list, test_types2_list, included_test_fnames = load_data(test_fnames, 
                                                                                                          tokenizer=tokenizer, 
                                                                                                          n_types=n_types,
                                                                                                          n_r_types=n_r_types,
                                                                                                          max_n_nodes=args.max_n_nodes_test,
                                                                                                          ignore_node_type=args.ignore_node_type, 
                                                                                                          ignore_node_name=args.ignore_node_name,
                                                                                                          ignore_title_node=args.ignore_title_node,
                                                                                                          is_test=True)
    n_test = len(test_features_list)
    included_docids = [fname.split('/')[-1].replace('.features', '').strip() for fname in included_test_fnames]
    with open('{}/predicted_ents_{}.txt'.format(args.model_output_dir, corpus_type), 'w') as fout_pred:
        compute_test(None, fout_pred, corpus_type, included_docids, docid2full)
