import argparse
import os
import logging
import shutil
from sklearn.model_selection import train_test_split
import json

def write_user_sets(root_dir, train_fnames, test_fnames, val_fnames):
    trees_directory = "produced_data/trees"
    train_user_ids = set()
    for fname in train_fnames:
        path = os.path.join(trees_directory, fname)
        with open(path) as f:
           tree_json = json.load(f)
        train_user_ids.update([node["user_id"] for node in tree_json["nodes"]])

    test_user_ids = set()
    for fname in test_fnames:
        path = os.path.join(trees_directory, fname)
        with open(path) as f:
           tree_json = json.load(f)
        test_user_ids.update([node["user_id"] for node in tree_json["nodes"]])

    val_user_ids = set()
    for fname in val_fnames:
        path = os.path.join(trees_directory, fname)
        with open(path) as f:
           tree_json = json.load(f)
        val_user_ids.update([node["user_id"] for node in tree_json["nodes"]])

    path = os.path.join(root_dir, "train_user_ids.json")
    with open(path, "w") as f:
        json.dump({
            "user_ids": list(train_user_ids)
        },
        f,
        indent=2)

    path = os.path.join(root_dir, "test_user_ids.json")
    with open(path, "w") as f:
        json.dump({
            "user_ids": list(test_user_ids)
        },
        f,
        indent=2)

    path = os.path.join(root_dir, "val_user_ids.json")
    with open(path, "w") as f:
        json.dump({
            "user_ids": list(val_user_ids)
        },
        f,
        indent=2)

def run(args):
    logging.info("Splitting trees to train test validation datasets")
    logging.info("Validation size:%s Test size:%s" % (args.val_size, args.test_size))

    root_dir = os.path.join("produced_data", "datasets", args.output_dir)
    logging.info("Writing files in %s directory" % root_dir)
    if os.path.exists(root_dir) and os.path.isdir(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)
    train_path = os.path.join(root_dir, "train")
    test_path = os.path.join(root_dir, "test")
    val_path = os.path.join(root_dir, "val")
    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(test_path)

    trees = os.listdir("produced_data/trees")
    train_fnames, test_fnames = train_test_split(trees, test_size=args.test_size, shuffle=True, random_state=1)
    train_fnames, val_fnames = train_test_split(train_fnames, test_size=args.val_size, shuffle=False, random_state=1)

    logging.info("The dataset has %s train trees" % (len(train_fnames)))
    logging.info("The dataset has %s val trees" % (len(val_fnames)))
    logging.info("The dataset has %s test trees" % (len(test_fnames)))

    for fname in train_fnames:
        shutil.copy("produced_data/trees/%s" % (fname), os.path.join(train_path, fname))
    for fname in test_fnames:
        shutil.copy("produced_data/trees/%s" % (fname), os.path.join(test_path, fname))
    for fname in val_fnames:
        shutil.copy("produced_data/trees/%s" % (fname), os.path.join(val_path, fname))

    logging.info("Writing sets of user ids for train, test, validation datasets")
    write_user_sets(root_dir, train_fnames, test_fnames, val_fnames)

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python tweets_to_trees.py")
    parser.add_argument(
        "--test-size",
        help="Test size",
        dest="test_size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--val-size",
        help="Validation size",
        dest="val_size",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory",
        dest="output_dir",
        type=str
    )

    args = parser.parse_args()
    run(args)