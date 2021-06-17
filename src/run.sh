set -e

# data preparation
mkdir ../data/json/train/processed/graph-sum
python generate_graph_sum_input.py train

mkdir ../data/json/valid/processed/graph-sum
python generate_graph_sum_input.py valid

mkdir ../data/json/test/processed/graph-sum
python generate_graph_sum_input.py test

mkdir ../data/json/auto_test/processed/graph-sum
python generate_graph_sum_input.py auto_test

# write gold graphs for evaluation
mkdir -p output_graphs/gold
python make_gold_graphs.py valid
python make_gold_graphs.py test
python make_gold_graphs.py auto_test

# training
python train.py

# inference
python test.py

# create output graphs
mkdir -p output_graphs/g2g
python make_predicted_graphs.py test output/predicted_ents_test.txt output_graphs/g2g/test
python make_predicted_graphs.py auto_test output/predicted_ents_auto_test.txt output_graphs/g2g/auto_test

# evaluate
python eval.py output_graphs/gold/test output_graphs/g2g/test test 0                 # test, untyped
python eval.py output_graphs/gold/test output_graphs/g2g/test test 1                 # test, typed
python eval.py output_graphs/gold/auto_test output_graphs/g2g/auto_test auto_test 0  # auto_test, untyped
python eval.py output_graphs/gold/auto_test output_graphs/g2g/auto_test auto_test 1  # auto_test, typed
