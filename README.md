# GraphSum


This repo contains the data and code for the G2G model in the paper: [Extracting Summary Knowledge Graphs from Long Documents](https://arxiv.org/abs/2009.09162). The other baseline TTG is simply based on [BertSumExt](https://github.com/nlpyang/PreSumm).


## Environment Setup
This code is tested on python 3.6.9, transformer 3.0.2 and pytorch 1.7.0. You would also need numpy and scipy packages.


## Data
Download and unzip the data from [this link](https://drive.google.com/file/d/1cGYHnsm6Frq4lp0pBTwU5xsQdx0wa5T8/view?usp=sharing). Put the unzipped folder named as `./data` parallel with `./src`. You should see four subfolders under `./data/json`, corresponding to four data splits as described in the paper. <br>

Under each subfolder, the json file contains all document full texts, abstracts as well as the summarized graphs obtained from the abstract, organized by the document keys. Each full text consists of a list of sections. Each summarized graph contains a list of entity and relation mentions. Except for the test split, other three data splits have their summarized graphs obtained by running [DyGIE++](https://github.com/dwadden/dygiepp) on the abstract. The test set have manually annotated summarized graphs from SciERC dataset. The format of the graph follows the output of DyGIE++, where each entity mention in a section is represented by (start token id, end token id, entity type) and each relation mention is represented by (start token id of entity 1, end token id of entity 1, start token id of entity 2, end token id of entity 2, relation type). The graph also contains a list of coreferential entity mentions. <br>

You should also see two subfolders under the `processed` folder of each data split: `merged_entities` and `aligned_entities`. `merged_entities` contains the full and summarized graphs for each document, where the graph vertices are cluster of entity mentions. Entity clusters in each summarized graph are coreferential entity mentions predicted by DyGIE++ or annotated (in test set). Entity clusters in each full graph contains entity mentions that are coreferences or share the same non-generic string names (as described in our paper). Under `merged_entities`, we provide entity clusters and relations between entity clusters, as well as corresponding entity and relation mentions in the full paper or abstract. Each relation is represented by "[entity cluster id 1]\_[entity cluster id 2]\_[relation type]". The original full graphs with all entity and relation mentions are obtained by running DyGIE++ on the document full text. You don't need them to run the code, but you can find them [here](https://drive.google.com/file/d/1g12gufjFgU--2BpABz0yP3KXgGRnHfNv/view?usp=sharing). For some entity names, you may see a trailing string "<GENERIC_ID> [number]". It means these entity names are classified by DyGIE++ as "generic" and the trailing string is used to differentiate the same entity name strings in different clusters in such cases. <br>

`aligned_entities` contains the pre-calculated alignment between entity clusters (see Section 5.1 in the paper) in the summarized and full graphs for each document. In each entity alignment file, under each entity cluster of the summarized graph, there is a list of entity clusters from the full graph if the list is not empty. They are used to facilitate data preprocessing of G2G and evaluation.


## Training and Evaluation

The model is based on [GAT](https://github.com/Diego999/pyGAT). Go to `./src` and run `bash run.sh`. You can also find the pretrained model [here](https://drive.google.com/file/d/1tSqgyaE9kHWHs-B-f-2F4vUN8Mhxm4uh/view?usp=sharing). Put it under `./src/output` and run the inference and evaluation parts in `./src/run.sh`.
