#!/usr/bin/env python

import argparse
import json
import time
import os
import jgrapht

from jgrapht.io.exporters import write_json

from models import Dataset
from dags import create_dags


def run(args):

    print('Loading dataset')
    dataset = Dataset()
    print('Loading users and followers from: {}'.format(args.input_followers_dir))
    dataset.load_users_and_followers(args.input_followers_dir)
    print('Loading tweets from: {}'.format(args.input_tweets_dir))
    dataset.load_tweets(args.input_tweets_dir)

    if not os.path.exists(args.output_dags_dir):
        os.makedirs(args.output_dags_dir)

    print('Writing dags to: {}'.format(args.output_dags_dir))
    for i, dag in enumerate(create_dags(dataset)):
        dag_path = os.path.join(args.output_dags_dir, 'dag-{}.json'.format(i))
        write_json(dag, dag_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        epilog="Example: python tweets_to_graph.py --input-dir raw_data --output-file tweets_graph.json"
    )
    parser.add_argument(
        "--input-tweets-dir",
        help="Input directory containing tweets as json files",
        dest="input_tweets_dir",
        type=str,
        default="raw_data/tweets"
    )
    parser.add_argument(
        "--input-followers-dir",
        help="Input directory containing user with followers as json files",
        dest="input_followers_dir",
        type=str,
        default="raw_data/followers"
    )
    parser.add_argument(
        "--output-dags-dir",
        help="Output directory to exports the dags",
        dest="output_dags_dir",
        type=str,
        default="raw_data/dags"
    )
    args = parser.parse_args()
    run(args)
