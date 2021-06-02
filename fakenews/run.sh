#!/usr/bin/env bash

rm -r produced_data/*
rm -r tweets1
rm -r trees2

echo "##### RUN dataset_preprocess #####"
python dataset_preprocess.py --ignore-dataset-pkl --sample-probability 0.01
echo "##### RUN create_trees #####"
python create_trees.py --tweets tweets1
mv trees2 produced_data/trees
echo "##### RUN generate_kfolds #####"
python generate_kfolds.py --k 10 --val-size 0.25
echo "##### RUN compute_user_labels #####"
python compute_user_labels.py --input-dir=../raw_data
echo "##### RUN users_to_graph #####"
for i in {0..9}
do
  echo "## dataset$i ##"
  python users_to_graph.py --input-dir ../raw_data --embeddings-file ../raw_data/glove.twitter.27B.100d.txt --dataset-root produced_data/datasets/dataset$i
done


echo Compressing data for local graphsage training...

./compress_data_for_graphsage_training.sh

read -p "Download the data locally, run the graphsage training, upload the results and then press [Enter] key to continue..."

echo "##### RUN compute_user_embeddings #####"
for i in {0..9}
do
  echo "## dataset$i ##"
  python compute_user_embeddings.py --input-dir ../raw_data --dataset-root produced_data/datasets/dataset$i --embeddings-file ../raw_data/glove.twitter.27B.100d.txt
done
echo "##### RUN add_trees_information #####"
for i in {0..9}
do
  echo "## dataset$i ##"
  python add_trees_information.py --dataset-root produced_data/datasets/dataset$i
done


echo Compressing data for local GAT training...

./compress_data_for_gat_training.sh

echo Download again the data locally. You can now run the final training of the model on your computer...
