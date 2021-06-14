import sys
import numpy as np
import json
import os
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from env import DATA_DIR


def validate_rels(graph):
    entities = graph['entities']
    relations = graph['relations']
    for e1, e2 in relations:
        assert e1 in entities
        assert e2 in entities


def _get_geid2peids(alignment_fname):
    geid2peids = defaultdict(set)
    with open(alignment_fname) as fin:
        for line in fin:
            if line.startswith('\t'):
                peid = line.strip().split('\t')[1][4:-1]
                geid2peids[geid].add(peid)
            elif line.startswith('EID'):
                geid = line.strip().split('\t')[0][4:-1]
            else:
                continue
    return geid2peids

""" get entity cluster type (type with highest frequency) """
def _get_cluster_id2type(cluster_id2pred_eids, pred_eids):
    cluster_id2type = {}
    for cid in cluster_id2pred_eids:
        type2freq = defaultdict(int)
        for eid in cluster_id2pred_eids[cid]:
            type2freq[pred_eids[eid][0]] += pred_eids[eid][2]
        best_type = sorted(type2freq.items(), key=lambda x: x[-1], reverse=True)[0][0]
        cluster_id2type[cid] = best_type
    return cluster_id2type

""" get relation type (all predicted types between each pairs in 2 entity clusters) """
def _get_rel_cluster2types(pred_rels_in_cluster_ids, pred_rels, cluster_id2pred_eids):
    rel_cluster2types = defaultdict(set)
    for r in pred_rels_in_cluster_ids:
        cid1, cid2 = r
        for eid1 in cluster_id2pred_eids[cid1]:
            for eid2 in cluster_id2pred_eids[cid2]:
                if (eid1, eid2) in pred_rels:
                    for rtype in pred_rels[(eid1, eid2)]:
                        rel_cluster2types[r].add(rtype)
    assert len(rel_cluster2types) == len(pred_rels_in_cluster_ids)
    return rel_cluster2types



def eval_graph_hard_align(gold_graph, pred_graph, docid, corpus_type, match_type=False):
    """ getting alignment map """
    assert corpus_type in ['valid', 'test', 'auto_test']
    geid2peids = _get_geid2peids('{}/json/{}/processed/aligned_entities/{}.txt'.format(DATA_DIR, corpus_type, docid))
    

    # each graph should be ents: {eid: (type, names, freq)} and rels: {(eid1, eid2): [type1, type2, ...]}
    gold_eids = gold_graph['entities']
    pred_eids = pred_graph['entities']
    gold_rels = gold_graph['relations']
    pred_rels = pred_graph['relations']

    

    """ assign each predicted node to its corresponding cluster """
    cluster_id2pred_eids = defaultdict(set)
    unaligned_cluster_id = 0
    pred_eid2cluster_id  = {}
    # for each predicted entity eid (node), find its alignment to target graph as the cluster (id as the eid in target graph)
    # for any unaligned node, it simply become a single node with cluster id as "unaligned_id" 
    for peid in pred_eids:
        aligned = False
        for geid in gold_eids:
            if peid in geid2peids[geid]:
                cluster_id2pred_eids[geid].add(peid)
                pred_eid2cluster_id[peid] = geid
                aligned = True
                break
        if not aligned:
            cluster_id = 'unaligned_{}'.format(unaligned_cluster_id)
            cluster_id2pred_eids[cluster_id].add(peid)
            pred_eid2cluster_id[peid] = cluster_id
            unaligned_cluster_id += 1

    """ assign entity type for each cluster """
    cluster_id2type = _get_cluster_id2type(cluster_id2pred_eids, pred_eids) # TODO


    """ count the number of target nodes being predicted """
    matched_eid_gold = 0
    for geid in gold_eids:
        if geid in cluster_id2type and (not match_type or gold_eids[geid][0] == cluster_id2type[geid]):
            matched_eid_gold += 1

    """ count the number of predicted entity clusters being correct """
    """ count total number of aligned clusters and entities in all clusters : for calculating entity dup rate """
    matched_eid_pred = 0
    ent_dup_rate = (0, 0)
    for cluster_id in cluster_id2pred_eids:
        if cluster_id in gold_eids:
            if not match_type:
                matched_eid_pred += 1
            else:
                if cluster_id2type[cluster_id] == gold_eids[cluster_id][0]:
                    matched_eid_pred += 1
            ent_dup_rate = (ent_dup_rate[0]+len(cluster_id2pred_eids[cluster_id]), ent_dup_rate[1]+1)
        else:
            assert len(cluster_id2pred_eids[cluster_id]) == 1
    # matched_eid_gold should equal matched_eid_pred
    assert matched_eid_pred == matched_eid_gold
    if ent_dup_rate[1] == 0:
        ent_dup_rate = 1.0
    else:
        ent_dup_rate = ent_dup_rate[0]*1.0/ent_dup_rate[1]


    """ represent predicted relations in cluster ids (relations between the same clusters with the same direction are merged) """
    pred_rels_in_cluster_ids = set()
    for peid1, peid2 in pred_rels:
        if not match_type:
            rel_in_cluster_ids = frozenset([pred_eid2cluster_id[peid1], pred_eid2cluster_id[peid2]])
        else:
            rel_in_cluster_ids = (pred_eid2cluster_id[peid1], pred_eid2cluster_id[peid2])
        pred_rels_in_cluster_ids.add(rel_in_cluster_ids)


    """ assign relation type(s) to each relation if match_type == True"""
    if match_type:
        rel_cluster2types = _get_rel_cluster2types(pred_rels_in_cluster_ids, pred_rels, cluster_id2pred_eids)


    """ count number of target relations being predicted """
    matched_rel_gold = 0
    gold_rels_without_type_and_direction = set()
    for geid1, geid2 in gold_rels:
        matched_found = False
        if not match_type:
            key = frozenset([geid1, geid2])
            # do not count multiple times for the same key
            if key in gold_rels_without_type_and_direction:
                continue
            gold_rels_without_type_and_direction.add(key)
            if key in pred_rels_in_cluster_ids:
                matched_rel_gold += 1
        else:
            key = (geid1, geid2)
            if key in rel_cluster2types:
                for gold_type in gold_rels[key]:
                    if gold_type in rel_cluster2types[key]:
                        matched_rel_gold += 1
    
    """ count number of predicted relations being correct """
    matched_rel_pred = 0
    for pred_rel in pred_rels_in_cluster_ids:
        if not match_type:
            if pred_rel in gold_rels_without_type_and_direction:
                matched_rel_pred += 1
        else:
            if pred_rel in gold_rels:
                for pred_type in rel_cluster2types[pred_rel]:
                    if pred_type in gold_rels[pred_rel]:
                        matched_rel_pred += 1
    assert matched_rel_pred ==matched_rel_gold



    matched_eid_gold = matched_eid_gold*1.0
    matched_eid_pred = matched_eid_pred*1.0
    matched_rel_gold = matched_rel_gold*1.0
    matched_rel_pred = matched_rel_pred*1.0


    prec_ent = matched_eid_pred/(len(cluster_id2pred_eids)+1e-6)
    recall_ent = matched_eid_gold/(len(gold_eids)+1e-6) 
    f1_ent = 2*(prec_ent*recall_ent)/(prec_ent+recall_ent+1e-6)

    if not match_type:
        prec_rel = matched_rel_pred/(len(pred_rels_in_cluster_ids)+1e-6)
        recall_rel = matched_rel_gold/(len(gold_rels_without_type_and_direction)+1e-6)
        f1_rel = 2*(prec_rel*recall_rel)/(prec_rel+recall_rel+1e-6)
    else:
        # count each (edge link, relation type) as 1
        total = sum([len(rel_cluster2types[r]) for r in rel_cluster2types])
        prec_rel = matched_rel_pred/(total+1e-6)
        recall_rel = matched_rel_gold/(sum([len(gold_rels[r]) for r in gold_rels])+1e-6)
        f1_rel = 2*(prec_rel*recall_rel)/(prec_rel+recall_rel+1e-6)

    res = {"entity" : {"precision":prec_ent, "recall":recall_ent, "f1":f1_ent}, "relation" : {"precision":prec_rel, "recall":recall_rel, "f1":f1_rel}, "ent_dup_rate":ent_dup_rate}
    return res

def get_entities_map(full_graph, pred_ents):
    eid2mentions = full_graph['entity_cluster_id_to_mentions']
    eid2names = full_graph['entity_cluster_id_to_string_names']
    eid2type = {}
    eid2freq = {}
    for eid in eid2mentions:
        mentions = eid2mentions[eid]
        eid2type[eid] = Counter([m[-1] for m in eid2mentions[eid]]).most_common(1)[0][0]
        eid2freq[eid] = len(eid2mentions[eid])
    results = {}
    for eid in pred_ents:
        if str(eid) not in eid2freq:
            continue
        results[str(eid)] = (eid2type[str(eid)], eid2names[str(eid)], eid2freq[str(eid)])
    return results

def find_valid_relations(full_graph, pred_ents):
    rels = full_graph['relations']
    rel2mentions = full_graph['relation_id_to_mentions']
    valid_eid_set = set(pred_ents.keys())
    valid_rels = defaultdict(set)
    for rid in rels:
        assert len(rels[rid]) == 1
        eid1, eid2, rtype = rels[rid][0].split('_')
        if eid1 in valid_eid_set and eid2 in valid_eid_set and eid1 != eid2:
            valid_rels[(eid1, eid2)].add(rtype)
    return valid_rels


def find_valid_relations_from_predicted_rels(predicted_rels, pred_ents, n_r_types, n_ents):
    assert n_r_types == 7
    rid2type = {0:'COMPARE', 1:'PART-OF', 2:'CONJUNCTION', 3:'EVALUATE-FOR', 4:'FEATURE-OF', 5:'USED-FOR', 6:'HYPONYM-OF'}

    predicted_rels_indices = (predicted_rels.max(1)[1].flatten() < n_r_types).nonzero().flatten().tolist()
    predicted_rels_types = predicted_rels.max(1)[1]
    valid_eid_set = set(pred_ents.keys())
    valid_rels = defaultdict(set)
    for eid1 in valid_eid_set:
        for eid2 in valid_eid_set:
            idx = int(eid1)*n_ents+int(eid2)
            if idx in predicted_rels_indices:
                rid = predicted_rels_types[idx]
                assert rid < n_r_types
                valid_rels[(eid1, eid2)].add(rid2type[int(rid)])

    return valid_rels


def get_docid2full(full_graph_fname):
    docid2full = {}
    with open(full_graph_fname) as f_full:
        for line in f_full:
            full_graph = json.loads(line.strip())
            docid = full_graph['doc_key']
            docid2full[docid] = full_graph
    return docid2full


def get_docid2graph(fname):
    docid2graph = {}
    with open(fname) as fin:
        all_docs = json.load(fin)
        for content in all_docs:
            graph = {'entities':{}, 'relations':{}}
            for eid in content['entities']:
                graph['entities'][eid] = content['entities'][eid]
            for rel in content['relations']:
                eid1, eid2 = rel.split('_')
                graph['relations'][(eid1, eid2)] = content['relations'][rel]
            docid2graph[content['doc_key']] = graph
    return docid2graph


# each graph should be ents: {eid: (type, names, freq)} and rels: {(eid1, eid2): [type1, type2, ...]}
def eval_graph(gold_graph, pred_graph, docid, corpus_type, match_type=False):
    validate_rels(gold_graph)
    validate_rels(pred_graph)
    edit_hard_results = eval_graph_hard_align(gold_graph, pred_graph, docid, corpus_type, match_type=match_type)

    return edit_hard_results


def eval(gold_fname, pred_fname, corpus_type, match_type):
    docid2gold = get_docid2graph(gold_fname)
    docid2pred = get_docid2graph(pred_fname)
    ave_edit_hard = {"entity" : {"precision":0.0, "recall":0.0, "f1":0.0}, "relation" : {"precision":0.0, "recall":0.0, "f1":0.0}, "ent_dup_rate":0.0}

    n_docs = len(docid2pred)
    for docid in docid2pred:
        edit_hard_results = eval_graph(docid2gold[docid], docid2pred[docid], docid, corpus_type, match_type=match_type)
        ave_edit_hard['entity']['precision'] += edit_hard_results['entity']['precision']
        ave_edit_hard['entity']['recall'] += edit_hard_results['entity']['recall']
        ave_edit_hard['entity']['f1'] += edit_hard_results['entity']['f1']
        ave_edit_hard['relation']['precision'] += edit_hard_results['relation']['precision']
        ave_edit_hard['relation']['recall'] += edit_hard_results['relation']['recall']
        ave_edit_hard['relation']['f1'] += edit_hard_results['relation']['f1']
        ave_edit_hard['ent_dup_rate'] += edit_hard_results['ent_dup_rate']
        new_edit_hard_results = dict(edit_hard_results)
        for key in edit_hard_results:
            if key == 'ent_dup_rate':
                new_edit_hard_results[key] = f"{edit_hard_results[key]:.3f}"
            else:
                for key2 in edit_hard_results[key]:
                    new_edit_hard_results[key][key2] = f"{edit_hard_results[key][key2]:.3f}"



    ave_edit_hard['entity']['precision'] /= n_docs
    ave_edit_hard['entity']['precision'] = round(ave_edit_hard['entity']['precision'], 4)

    ave_edit_hard['entity']['recall'] /= n_docs
    ave_edit_hard['entity']['recall'] = round(ave_edit_hard['entity']['recall'], 4)

    ave_edit_hard['entity']['f1'] /= n_docs
    ave_edit_hard['entity']['f1'] = round(ave_edit_hard['entity']['f1'], 4)

    ave_edit_hard['relation']['precision'] /= n_docs
    ave_edit_hard['relation']['precision'] = round(ave_edit_hard['relation']['precision'], 4)

    ave_edit_hard['relation']['recall'] /= n_docs
    ave_edit_hard['relation']['recall'] = round(ave_edit_hard['relation']['recall'], 4)

    ave_edit_hard['relation']['f1'] /= n_docs
    ave_edit_hard['relation']['f1'] = round(ave_edit_hard['relation']['f1'], 4)

    ave_edit_hard['ent_dup_rate'] /= n_docs
    ave_edit_hard['ent_dup_rate'] = round(ave_edit_hard['ent_dup_rate'], 4)



    print('========== EVALUATION RESULTS : ===========')
    print(ave_edit_hard)



if __name__ == "__main__":
    gold_graph_fname = sys.argv[1]
    pred_graph_fname = sys.argv[2]
    corpus_type = sys.argv[3]
    assert corpus_type in ['valid', 'test', 'auto_test']
    typed_eval = sys.argv[4]=='1'

    eval(gold_graph_fname, pred_graph_fname, corpus_type, typed_eval)
