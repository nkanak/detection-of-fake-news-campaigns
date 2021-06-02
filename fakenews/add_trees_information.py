#!/usr/bin/env python

#
# Tweets to dags version for FakeNews
#

import argparse
import json
import os
import logging
from tqdm import tqdm

def run(args):

    users_embeddings_root = os.path.join(args.dataset_root, "user_embeddings")
    users_embeddings_files = os.listdir(users_embeddings_root)
    with open(os.path.join(users_embeddings_root, users_embeddings_files[0])) as f:
        not_found_embedding = [0]*len(json.load(f)['embedding'])
    users_embeddings_files = set(users_embeddings_files)

    train_trees_path = os.path.join(args.dataset_root, "train")
    val_trees_path = os.path.join(args.dataset_root, "val")
    test_trees_path = os.path.join(args.dataset_root, "test")

    for path, tree_filenames in [(train_trees_path, os.listdir(train_trees_path)), (val_trees_path, os.listdir(val_trees_path)), (test_trees_path, os.listdir(test_trees_path))]:
        logging.info("Adding node embeddings to trees %s" % (path))
        for fname in tqdm(tree_filenames):
            with open(os.path.join(path, fname)) as f:
                tree = json.load(f)
            for i, node in enumerate(tree['nodes']):
                if not '%s.json' % (node['user_id']) in users_embeddings_files:
                    tree['nodes'][i]['embedding'] = not_found_embedding
                else:
                    with open(os.path.join(users_embeddings_root, '%s.json' % (node['user_id']))) as f:
                        user_embedding = json.load(f)
                    tree['nodes'][i]['embedding'] = user_embedding["embedding"]
            with open(os.path.join(path, fname), 'w') as f:
                json.dump(tree, f)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python tweets_to_dags.py")

    parser.add_argument(
        "--dataset-root",
        help="Dataset path",
        dest="dataset_root",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    run(args)
