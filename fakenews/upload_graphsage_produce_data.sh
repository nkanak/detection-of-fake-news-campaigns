#!/usr/bin/env bash

cd produced_data
for i in {0..9}
do
  echo Connecting to the server
  scp dataset$i/users_graphsage_embeddings_lookup.json astroturfer@10.100.54.29:/home/astroturfer/astroturfing/fakenews/produced_data/datasets/dataset$i/
done
