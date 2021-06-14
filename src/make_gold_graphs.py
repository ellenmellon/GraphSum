import sys
import json
from collections import Counter, defaultdict
from env import DATA_DIR, GRAPH_DIR

def find_entities(full_graph):
    eid2mentions = full_graph['entity_cluster_id_to_mentions']
    eid2names = full_graph['entity_cluster_id_to_string_names']
    eid2freq = {}
    eid2type = {}
    for eid in eid2mentions:
        eid2freq[eid] = len(eid2mentions[eid])
        eid2type[eid] = Counter([m[-1] for m in eid2mentions[eid]]).most_common(1)[0][0]
    
    results = {}
    freq_ents = sorted(eid2freq.items(), key=lambda x: x[1], reverse=True)
    for eid in sorted([e[0] for e in freq_ents]):
        results[eid] = (eid2type[eid], eid2names[eid], eid2freq[eid])
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
fname_graph = '{}/json/{}/processed/merged_entities/target_graph.json.ent_merged'.format(DATA_DIR, corpus_type)
result_fname = '{}/gold/{}'.format(GRAPH_DIR, corpus_type)


preds = []
with open(fname_graph) as f_graph, open(result_fname, 'w') as fout:
    for line in f_graph:
        graph = json.loads(line.strip())
        docid = graph['doc_key']

        pred = {}
        pred['doc_key'] = docid
        eid2info = find_entities(graph)
        valid_rels = find_valid_relations(graph, eid2info)
        
        new_eid2info = {}
        for rel in valid_rels:
            eid1, eid2 = rel.split('_')
            new_eid2info[eid1] = eid2info[eid1]
            new_eid2info[eid2] = eid2info[eid2]
        pred['entities'] = new_eid2info
        pred['relations'] = valid_rels

        preds += [pred]
    fout.write(json.dumps(preds, indent=2, sort_keys=True))
