#!/usr/bin/env bash

cd produced_data
echo Compressing user labels
tar -czf user_labels.tar.gz user_labels

echo Copying files needed for graphsage to the new directory 
for i in {0..9}
do
  rm -r dataset$i
  mkdir  dataset$i
  cp -r datasets/dataset$i/train_edges.pkl datasets/dataset$i/train_vertices.pkl  \
        datasets/dataset$i/val_edges.pkl  datasets/dataset$i/val_vertices.pkl  datasets/dataset$i/test_edges.pkl  datasets/dataset$i/test_vertices.pkl dataset$i/
done

echo Compressing the new directories
for i in {0..9}
do
  tar -czf graphsage_dataset$i.tar.gz dataset$i
done
