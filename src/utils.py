import os
import numpy as np
import scipy.sparse as sp
import torch
import time

CLASSES = ['not salient', 'salient']

def encode_onehot(labels):
    classes = CLASSES # make this consistent for each graph
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def get_all_feature_fnames(path):
    dir_list = os.listdir(path)
    fnames = []
    for f in dir_list:
        if not f.endswith('.features'):
            continue
        fnames += ['{}/{}'.format(path, f)]
    return fnames


def load_features_type_tokens_label(fname, tokenizer, ignore_title_node, max_n_nodes=-1):
    features_row = []
    features_col = []
    features_data = []
    type_row = []
    type_col = []
    type_data = []
    token_ids = []
    type2_row = []
    type2_col = []
    type2_data = []
    idx = []
    labels = []
    with open(fname) as fin:
        if ignore_title_node:
            # if ignore title node (first line in the file), start from 2nd line
            row_id = -2
        else:
            row_id = -1
        for line in fin:
            row_id += 1
            if row_id < 0:
                continue
            if max_n_nodes > 0 and row_id >= max_n_nodes:
                break
            content = line.strip().split('\t')
            ind = int(content[0])
            label = content[-1]
            features_row += [row_id]
            features_col += [0]
            features_data += [int(content[2])]
            type2_row += [row_id]
            type2_data += [1]
            type2_col += [min([int(content[3]), 19])]
            type_row += [row_id]
            type_col += [int(content[1])]
            type_data += [1]
            if ignore_title_node:
                # hard-coded max length of name sequence
                token_ids += [tokenizer.encode(content[-2])[:10]]
                token_ids[-1] = token_ids[-1] + (10-len(token_ids[-1]))*[tokenizer.pad_token_id]
            else:
                token_ids += [tokenizer.encode(content[-2])[:30]]
                token_ids[-1] = token_ids[-1] + (30-len(token_ids[-1]))*[tokenizer.pad_token_id]
            idx += [ind]
            labels += [label]
    idx = np.array(idx, dtype=np.int32)
    f_row = np.array(features_row)
    f_col = np.array(features_col)
    f_data = np.array(features_data, dtype=np.float32)
    t_row = np.array(type_row)
    t_col = np.array(type_col)
    t_data = np.array(type_data)
    t2_row = np.array(type2_row)
    t2_col = np.array(type2_col)
    t2_data = np.array(type2_data)
    token_ids = np.array(token_ids, dtype=np.float32)
    labels = np.array(labels, dtype=np.dtype(str))
    return idx, f_row, f_col, f_data, t_row, t_col, t_data, t2_row, t2_col, t2_data,  token_ids, labels


def load_edges_by_rtype(fname, idx_map, n_r_types, ignore_title_node, max_n_nodes=-1):
    if not ignore_title_node:
        n_r_types += 1
    data = [[] for _ in range(n_r_types)]
    row = [[] for _ in range(n_r_types)]
    col = [[] for _ in range(n_r_types)]

    with open(fname) as fin:
        row_id = -1
        for line in fin:
            row_id += 1
            eid1, eid2, r_idx, freq = line.strip().split('\t')
            if max_n_nodes > 0 and (int(eid1) >= max_n_nodes or int(eid2) >= max_n_nodes):
                continue
            idx = int(r_idx)
            data[idx] += [int(freq)]
            row[idx] += [idx_map[int(eid1)]]
            col[idx] += [idx_map[int(eid2)]]

    if not ignore_title_node:
        assert len(data[-1]) == 0
        data[-1] += [1]*len(idx_map)
        row[-1] += [idx_map[-1]]*len(idx_map)
        col[-1] += [idx_map[i] for i in idx_map]

    data_list = [np.array(d) for d in data]
    row_list = [np.array(r) for r in row]
    col_list = [np.array(c) for c in col]
    return data_list, row_list, col_list, row_id



def load_edge_labels(fname_orig, fname, e_labels, idx_map, n_r_types, max_n_nodes=-1):
    # no not consider title node for now ... so n_r_types should be 7 only
    assert n_r_types == 7
    labels = [n_r_types+1]*(len(idx_map)*len(idx_map))
    salient_eids = set(e_labels.nonzero()[0])
    with open(fname) as fin:
        for line in fin:
            eid1, eid2, rid = line.strip().split('\t')
            if max_n_nodes > 0 and (int(eid1) >= max_n_nodes or int(eid2) >= max_n_nodes):
                continue
            idx = idx_map[int(eid1)]*len(idx_map)+idx_map[int(eid2)]
            #labels[idx] = int(rid)
    with open(fname_orig) as fin:
        for line in fin:
            eid1, eid2, rid, freq = line.strip().split('\t')
            if int(eid1) in salient_eids and int(eid2) in salient_eids:
                idx = idx_map[int(eid1)]*len(idx_map)+idx_map[int(eid2)]
                #if labels[idx] == n_r_types+1:
                labels[idx] = min(int(rid), labels[idx])
    return np.array(labels)




def load_data(fnames, tokenizer=None, n_types=-1, n_r_types=-1, max_n_nodes=-1, ignore_node_type=False, ignore_node_name=False, ignore_title_node=False, is_test=False):
    """Each line : <node_id> <type> <features> <name_string>"""

    #print('Loading dataset ...')

    adj_list = []
    features_list = []
    labels_list = []
    r_labels_list = []
    token_ids_list = []
    types_list = []
    types2_list = []
    processed = 0
    start_time = time.time()
    included_fnames = []
    for fname in fnames:

        processed += 1
        assert fname.endswith('.features')
        if is_test:
            idx, f_row, f_col, f_data, t_row, t_col, t_data, t2_row, t2_col, t2_data, token_ids, labels = load_features_type_tokens_label(fname, tokenizer, ignore_title_node, max_n_nodes=max_n_nodes)
        else:
            idx, f_row, f_col, f_data, t_row, t_col, t_data, t2_row, t2_col, t2_data, token_ids, labels = load_features_type_tokens_label(fname, tokenizer, ignore_title_node)

        n_nodes = len(idx)


        # if in test mode, do not skip due to max_n_nodes
        if is_test:
            assert max_n_nodes < 0 or n_nodes <= max_n_nodes

        if max_n_nodes > 0 and n_nodes > max_n_nodes:
            continue
        if n_nodes == 0:
            continue

        features = sp.csr_matrix((f_data, (f_row, f_col)), shape=(n_nodes, 1))
        types = sp.csr_matrix((t_data, (t_row, t_col)), shape=(n_nodes, n_types))
        # hard-code # section ids
        types2 = sp.csr_matrix((t2_data, (t2_row, t2_col)), shape=(n_nodes, 20))
        labels = encode_onehot(labels)

        idx_map = {j: i for i, j in enumerate(idx)}
        if is_test:
            r_labels = load_edge_labels(fname.replace('.features', '.edges'), fname.replace('.features', '.edge_labels'), np.where(labels)[1], idx_map, n_r_types, max_n_nodes=max_n_nodes)
        else:
            r_labels = load_edge_labels(fname.replace('.features', '.edges'), fname.replace('.features', '.edge_labels'), np.where(labels)[1], idx_map, n_r_types)

        # build graph
        #idx_map = {j: i for i, j in enumerate(idx)}
        if is_test:
            data_list, row_list, col_list, n_edges = load_edges_by_rtype(fname.replace('.features', '.edges'), idx_map, n_r_types, ignore_title_node, max_n_nodes=max_n_nodes)
        else:
            data_list, row_list, col_list, n_edges = load_edges_by_rtype(fname.replace('.features', '.edges'), idx_map, n_r_types, ignore_title_node)
        try:
            adj = []
            for i in range(len(data_list)):
                cur_adj = sp.coo_matrix((data_list[i], (row_list[i], col_list[i])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
                # build symmetric adjacency matrix
                #cur_adj = cur_adj + cur_adj.T.multiply(cur_adj.T > cur_adj) - cur_adj.multiply(cur_adj.T > cur_adj)

                # normalization
                # TODO: enable the option  of not making adj symmentric ...
                #cur_adj = normalize_adj(cur_adj + sp.eye(cur_adj.shape[0]))
                #cur_adj = normalize_adj(cur_adj)
                #adj = torch.FloatTensor(np.array(adj.todense()))
                adj += [np.array(cur_adj.todense())]
            adj = torch.FloatTensor(np.array(adj))

        except:
            print('Exception!!!')
            print(labels.shape)
            continue


        
        features = torch.FloatTensor(np.array(features.todense()))
        types = torch.FloatTensor(np.array(types.todense()))
        types2 = torch.FloatTensor(np.array(types2.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        r_labels = torch.LongTensor(r_labels)
        token_ids = torch.LongTensor(token_ids)
        adj_list += [adj]
        included_fnames += [fname]
        

        features_list += [features]
        labels_list += [labels]
        r_labels_list += [r_labels]
        if ignore_node_name:
            token_ids_list += [None]
        else:
            token_ids_list += [token_ids]
        if ignore_node_type:
            types_list += [None]
        else:
            types_list += [types]
            types2_list += [types2]

    # return a list of adj, features, labels
    return adj_list, features_list, labels_list, r_labels_list, token_ids_list, types_list, types2_list, included_fnames


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

