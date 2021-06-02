#!/usr/bin/env bash

cd produced_data


echo Connecting to the server
scp  -T astroturfer@10.100.54.29:"/home/astroturfer/astroturfing/fakenews/produced_data/graphsage_dataset0.tar.gz \
                                  /home/astroturfer/astroturfing/fakenews/produced_data/graphsage_dataset1.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/graphsage_dataset2.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/graphsage_dataset3.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/graphsage_dataset4.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/graphsage_dataset5.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/graphsage_dataset6.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/graphsage_dataset7.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/graphsage_dataset8.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/graphsage_dataset9.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/user_labels.tar.gz" . 
								  

echo Uncompressing user_labels file
rm -r user_labels
tar -xzf user_labels.tar.gz -C .


echo Uncompressing the downloaded files
for i in {0..9}
do
  rm -r dataset$i
  tar -xzf graphsage_dataset$i.tar.gz -C .
done
