import sys
import json
import os
from collections import defaultdict, Counter
from env import DATA_DIR


type2id = {'Task':0, 'Method':1, 'Metric':2, 'Material':3, 'OtherScientificTerm':4, 'Generic':5, 'Title':6}

r_type2id = {'COMPARE':0, 'PART-OF':1, 'CONJUNCTION':2, 'EVALUATE-FOR':3, 'FEATURE-OF':4, 'USED-FOR':5, 'HYPONYM-OF':6, 'CoRef':7}

def read_idf(fname='{}/json/idf.txt'.format(DATA_DIR)):
    w2idf = {}
    with open(fname) as fin:
      for line in fin:
        w, idf = line.strip().split('\t')
        w2idf[w] = float(idf)
    return w2idf

W2IDF = read_idf()


def get_name_with_highest_idf(names):
    if len(names) == 1:
        return names[0].lower()
    highest_idf = -1
    best_name = None
    for n in names:
        idf = sum([W2IDF[t] if t in W2IDF else 0.0 for t in n.lower().split()])
        if idf > highest_idf:
            highest_idf = idf
            best_name = n
    return best_name.lower()


def resolve_eid_name(names):
    length2names = defaultdict(list)
    for n in names:
        name = n.split('<GENERIC_ID>')[0].strip()
        l = len(name.split())
        length2names[l].append(name)
    return get_name_with_highest_idf(length2names[sorted(length2names.keys(), reverse=True)[0]])


def create_edges_file(rels, out_fname):
    with open(out_fname, 'w') as fout:
        for i in rels:
            eid1, eid2, r_type, freq = rels[i]
            fout.write('{}\t{}\t{}\t{}\n'.format(eid1, eid2, r_type2id[r_type], freq))
            if r_type in ['COMPARE', 'CONJUNCTION']: # handle symmetric types 
                fout.write('{}\t{}\t{}\t{}\n'.format(eid2, eid1, r_type2id[r_type], freq))


def create_features_file(title, eid2name, eid2type, eid2freq, eid2sectionid, salient_eids, out_fname):
    with open(out_fname, 'w') as fout:
        # title line
        fout.write('\t'.join(['-1']+['6']+['0']+['0']+[title]+['not salient'])+'\n')
        for eid in eid2name:
            name = eid2name[eid]

            if eid in salient_eids:
                fout.write('\t'.join([eid]+[str(type2id[eid2type[eid]])]+[str(eid2freq[eid])]+[str(eid2sectionid[eid])]+[name]+['salient'])+'\n')
            else:
                fout.write('\t'.join([eid]+[str(type2id[eid2type[eid]])]+[str(eid2freq[eid])]+[str(eid2sectionid[eid])]+[name]+['not salient'])+'\n')


def create_edge_labels_file(target_rels, geid2peids, out_fname):
    with open(out_fname, 'w') as fout:
        for rid in target_rels:
            eid1, eid2, r_type = target_rels[rid]
            if eid1 not in geid2peids or eid2 not in geid2peids:
                continue
            peids_1 = geid2peids[eid1]
            peids_2 = geid2peids[eid2]
            for peid1 in peids_1:
                for peid2 in peids_2:
                    fout.write('\t'.join([peid1, peid2, str(r_type2id[r_type])])+'\n')
                    if r_type in ['COMPARE', 'CONJUNCTION']: # handle symmetric types 
                        fout.write('\t'.join([peid2, peid1, str(r_type2id[r_type])])+'\n')

        for eid in geid2peids:
            peids = geid2peids[eid]
            for peid1 in peids:
                for peid2 in peids:
                    if peid1 ==  peid2:
                        continue
                    fout.write('\t'.join([peid1, peid2, str(r_type2id['CoRef'])])+'\n')


def read_salient_and_coref_eids(aligned_ent_dir):
    print('reading salient and coref eids ...')
    docid2geid2peids = {}
    docid2salient_eids = {}
    ct = 0
    for file in os.listdir(aligned_ent_dir):
        ct += 1
        if ct%10000 == 0:
            print(f'read {ct} documents')
        fname = '{}/{}'.format(aligned_ent_dir, file)
        docid = '.'.join(file.split('.')[:-1])
        geid2peids = defaultdict(set)
        salient_eids = set()
        with open(fname) as fin:
            for line in fin:
                if line.startswith('\t'):
                    peid = line.strip().split('\t')[1][4:-1]
                    geid2peids[geid].add(peid)
                    salient_eids.add(peid)
                elif line.startswith('EID'):
                    geid = line.strip().split('\t')[0][4:-1]
                else:
                    continue
        docid2salient_eids[docid] = salient_eids
        docid2geid2peids[docid] =  geid2peids
    return docid2salient_eids, docid2geid2peids


def read_rels_and_eid2name(graph_fname):
    print('reading rels and eid2name ...')
    docid2eid2name, docid2rels = {}, {}
    docid2eid2freq, docid2eid2type = {}, {}
    docid2eid2sectionid = {}
    with open(graph_fname) as fin:
        ct = 0
        for line in fin:
            ct += 1
            if ct%10000 == 0:
                print(f'read {ct} full graphs')
            graph = json.loads(line.strip())
            docid = graph['doc_key']

            eid2names = graph['entity_cluster_id_to_string_names']
            eid2mentions = graph['entity_cluster_id_to_mentions']
            rid2mentions = graph['relation_id_to_mentions']
            eid2name = {}
            eid2freq = {}
            eid2type = {}
            eid2sectionid = {}
            for eid in eid2names:
                eid2freq[eid] = len(eid2mentions[eid])
                eid2sectionid[eid] = min([m[0] for m in eid2mentions[eid]])
                eid2type[eid] = Counter([m[-1] for m in eid2mentions[eid]]).most_common(1)[0][0]

                eid2name[eid] = resolve_eid_name(eid2names[eid])
            rels = graph['relations']
            new_rels = {}
            for rid in rels:
                eid1, eid2, r_type = rels[rid][0].split('_')
                new_rels[rid] = (eid1, eid2, r_type, len(rid2mentions[rid]))
            docid2eid2name[docid] = eid2name
            docid2rels[docid] = new_rels
            docid2eid2freq[docid] = eid2freq
            docid2eid2type[docid] = eid2type
            docid2eid2sectionid[docid] = eid2sectionid
    return docid2eid2name, docid2rels, docid2eid2freq, docid2eid2type, docid2eid2sectionid


def read_docid2target_rels(target_graph_fname):
    print('reading target rels ...')
    docid2target_rels = {}
    with open(target_graph_fname) as fin:
        ct = 0
        for line in fin:
            ct += 1
            if ct%10000 == 0:
                print(f'read {ct} target graphs')
            graph = json.loads(line.strip())
            docid = graph['doc_key']
            rels = graph['relations']
            new_rels = {}
            for rid in rels:
                eid1, eid2, r_type = rels[rid][0].split('_')
                new_rels[rid] = (eid1, eid2, r_type)
            docid2target_rels[docid] = new_rels
    return docid2target_rels



def read_docid2title(fname):
    docid2title = {}
    with open(fname) as fin:
        for line in fin:
            content = json.loads(line.strip())
            docid = content['doc_key']
            title = content['title'].lower()
            docid2title[docid] = title
    return docid2title


corpus_type = sys.argv[1]
assert corpus_type in ['valid', 'test', 'auto_test', 'train']
aligned_ent_dir = '{}/json/{}/processed/aligned_entities'.format(DATA_DIR, corpus_type)
graph_fname = '{}/json/{}/processed/merged_entities/full_graph.json.ent_merged'.format(DATA_DIR, corpus_type)
target_graph_fname = '{}/json/{}/processed/merged_entities/target_graph.json.ent_merged'.format(DATA_DIR, corpus_type)
if corpus_type == 'test':
    title_fname = '{}/json/test/titles.json'.format(DATA_DIR)
else:
    title_fname = '{}/json/train/titles.json'.format(DATA_DIR)

out_dir = '{}/json/{}/processed/graph-sum'.format(DATA_DIR, corpus_type)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


docid2salient_eids, docid2geid2peids  = read_salient_and_coref_eids(aligned_ent_dir)
docid2eid2name, docid2rels, docid2eid2freq, docid2eid2type, docid2eid2sectionid = read_rels_and_eid2name(graph_fname)
docid2title = read_docid2title(title_fname)
docid2target_rels = read_docid2target_rels(target_graph_fname)

print('start creating gat-graphsum input files ...')
ct = 0

for docid in docid2eid2name:
    ct += 1
    if ct%10000 == 0:
        print(f'processed {ct} documents')
    title = docid2title[docid]
    rels = docid2rels[docid]
    target_rels = docid2target_rels[docid]
    eid2name = docid2eid2name[docid]
    eid2freq = docid2eid2freq[docid]
    eid2sectionid = docid2eid2sectionid[docid]
    eid2type = docid2eid2type[docid]
    geid2peids = docid2geid2peids[docid]
    salient_eids = docid2salient_eids[docid]
    edge_fname = '{}/{}.edges'.format(out_dir, docid)
    edge_labels_fname = '{}/{}.edge_labels'.format(out_dir, docid)
    feature_fname = '{}/{}.features'.format(out_dir, docid)
    create_edges_file(rels, edge_fname)
    create_features_file(title, eid2name, eid2type, eid2freq, eid2sectionid, salient_eids, feature_fname)
    create_edge_labels_file(target_rels, geid2peids, edge_labels_fname)




