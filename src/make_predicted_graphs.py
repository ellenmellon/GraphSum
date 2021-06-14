import sys
import json
from collections import Counter, defaultdict
from env import DATA_DIR


def find_entities(full_graph, pred_ents):
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

def find_valid_relations(full_graph, freq_eids):
    rels = full_graph['relations']
    rel2mentions = full_graph['relation_id_to_mentions']
    valid_eid_set = set(freq_eids.keys())
    valid_rels = defaultdict(set)
    for rid in rels:
        assert len(rels[rid]) == 1
        eid1, eid2, rtype = rels[rid][0].split('_')
        if eid1 in valid_eid_set and eid2 in valid_eid_set and eid1 != eid2:
            valid_rels['{}_{}'.format(eid1, eid2)].add(rtype)
    for key in valid_rels:
        valid_rels[key] = list(valid_rels[key])
    return valid_rels


corpus_type = sys.argv[1]
fname_full = '{}/json/{}/processed/merged_entities/full_graph.json.ent_merged'.format(DATA_DIR, corpus_type)
predicted_ents_fname = sys.argv[2]
graph_result_fname = sys.argv[3]


preds = []
docid2pred_ents = {}
docid2relations = {}
with open(predicted_ents_fname) as fin:
    for line in fin:
        content = json.loads(line.strip())
        docid = content['doc_key']
        pred_ents = content['eids']
        if 'relations' in content:
            relations = content['relations']
        else:
            relations = None
        docid2pred_ents[docid] = pred_ents
        # TODO: change this to list of rel types
        docid2relations[docid] = relations

with open(fname_full) as f_full, open(graph_result_fname, 'w') as fout:
    for line in f_full:
        full_graph = json.loads(line.strip())
        docid = full_graph['doc_key']

        pred = {}
        pred['doc_key'] = docid
        if docid not in docid2pred_ents:
            continue
        pred_ents = docid2pred_ents[docid]
        eid2info = find_entities(full_graph, pred_ents)
        pred['entities'] = eid2info

        valid_rels = find_valid_relations(full_graph, eid2info)
        if docid2relations[docid] is None:
            pred['relations'] = valid_rels
        else:
            pred['relations'] = docid2relations[docid]

        preds += [pred]
    fout.write(json.dumps(preds, indent=2, sort_keys=True))
