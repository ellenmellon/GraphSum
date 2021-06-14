# GraphSum


This repo contains the data and code for the G2G model in the paper: [Extracting Summary Knowledge Graphs from Long Documents](https://arxiv.org/abs/2009.09162). The other baseline TTG is simply based on [BertSumExt](https://github.com/nlpyang/PreSumm).


## Environment Setup
This code is tested on python 3.6.9 and transformer 3.0.2. You would also need to install numpy and scipy packages.


## Data
Download and unzip the data from [this link](). Put the unzipped folder named as `./data` parallel with `./src`. You should see four subfolders under `./data/json`, corresponding to four data splits as described in the paper. <br>

Under each subfolder, the json file contains all document full texts as well as the summarized graphs obtained from the abstract, organized by the document keys. Except for the test split, other three data splits have their summarized graphs obtained by running [DyGIE++](https://github.com/dwadden/dygiepp) on the abstract. The test set have manually annotated summarized graphs from SciERC dataset.  <br>

You should also see two subfolders under the `processed` folder of each data split: `merged_entities` and `aligned_entities`. `merged_entities` contains entity clusters of the full and summarized graphs of each document, as well as the relations between entity clusters. The full graph is obtained by running DyGIE++ on the document full text. How to merge entities into clusters is described in the paper. For some entity names, you may see a trailing string "<GENERIC_ID> [number]". It means these entity names are classified by DyGIE++ as "generic" and the trailing string is used to differentiate the same entity name strings in different clusters in such cases. `aligned_entities` contains the alignment between entity clusters in the summarized and full graphs for each document. In each entity alignment file, under each entity cluster of the summarized graph, there is a list of entity clusters from the full graph if the list is not empty. The alignment algorithm is described in the paper.


## Training and Evaluation

The model is based on [GAT](https://github.com/Diego999/pyGAT). Go to `./src` and run `bash run.sh`. You can also find the pretrained model [here](https://drive.google.com/file/d/1tSqgyaE9kHWHs-B-f-2F4vUN8Mhxm4uh/view?usp=sharing). Put it under `./src/output` and run the inference and evaluation parts in `./src/run.sh`.
