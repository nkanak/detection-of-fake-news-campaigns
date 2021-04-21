# Astroturfing project - FakeNews dataset


Unzip all data into folder `../raw_data`. It should contain the following directory 
structure. 

```
politifact
user_followers
user_profiles
```

# Compute user embeddings

Download user embeddings in the root folder of the problem. 

```
curl -LO https://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```

Compute using 

```
./compute_user_embeddings.py --input-dir=../raw_data --output-dir=../raw_data --embeddings-file=../glove.twitter.27B.100d.txt
```

you should get a directory `../raw_data/user_embeddings` containing one json file per user. 

## Compute user labels 

Compute using 

```
./compute_user_labels.py --input-dir=../raw_data --output-dir=../raw_data
```

you should get a directory `../raw_data/user_labels` containing one json file per user. 


