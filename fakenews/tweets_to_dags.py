#!/usr/bin/env python

#
# Tweets to dags version for FakeNews
#

import argparse
import json
import time
import os
import jgrapht
import logging
import utils

from jgrapht.io.exporters import write_json

from dataset import FakeNewsDataset
from dags import create_dags


def run(args):

    logging.info("Loading dataset")

    dataset_pkl = "tweets-to-dags-dataset.pkl"
    if os.path.exists(dataset_pkl):
        logging.info("Loading dataset from: {}".format(dataset_pkl))
        dataset = utils.read_pickle_from_file(dataset_pkl)
    else:
        user_profiles_path = "{}/user_profiles".format(args.input_dir)
        user_followers_path = "{}/user_followers".format(args.input_dir)
        real_news_retweets_path = "{}/{}/real".format(args.input_dir, args.website)
        fake_news_retweets_path = "{}/{}/fake".format(args.input_dir, args.website)
        dataset = FakeNewsDataset(
            user_profiles_path=user_profiles_path,
            user_followers_path=user_followers_path,
            real_news_retweets_path=real_news_retweets_path,
            fake_news_retweets_path=fake_news_retweets_path,
        )
        dataset.load()
        utils.write_object_to_pickle_file(dataset_pkl, dataset)

    dags_path = "{}/dags".format(args.output_dir)
    logging.info("Writing dags to: {}".format(dags_path))
    os.makedirs(dags_path, exist_ok=True)
    for i, dag in enumerate(create_dags(dataset)):
        dag_path = os.path.join(dags_path, "dag-{}.json".format(i))
        write_json(dag, dag_path)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python tweets_to_dags.py")
    parser.add_argument(
        "--input-dir",
        help="Input directory containing the fakenewsnet dataset",
        dest="input_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--website",
        help="Either politifact or gossipcop",
        dest="website",
        type=str,
        default="politifact",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory to exports the dags",
        dest="output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    run(args)
