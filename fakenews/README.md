# Astroturfing project - FakeNews dataset


Unzip all data into folder `../raw_data`. It should contain the following directory 
structure. 

```
politifact
user_followers
user_profiles
```

## Compute user labels 

Compute using 

```
./compute_user_labels.py --input-dir=../raw_data --output-dir=../raw_data
```

you should get a directory `../raw_data/user_labels` containing one json file per user. 


