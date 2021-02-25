#!/usr/bin/env python

import argparse
import json
import time
import os
import jgrapht
import logging

from jgrapht.io.exporters import write_json

from datasets import Dataset
from dags import create_dags


def run(args):

    logging.info('Loading dataset')
    dataset = Dataset()
    logging.info('Loading users from: {}'.format(args.input_users_dir))
    dataset.load_users(args.input_users_dir)
    logging.info('Loading followers from: {}'.format(args.input_followers_dir))
    dataset.load_followers(args.input_followers_dir)
    if args.botometer:
        logging.info('Loading botometer data from: {}'.format(args.input_botometer_dir))
        dataset.load_botometer(args.input_botometer_dir)
    logging.info('Loading tweets from: {}'.format(args.input_tweets_dir))
    dataset.load_tweets(args.input_tweets_dir)

    if not os.path.exists(args.output_dags_dir):
        os.makedirs(args.output_dags_dir)

    logging.info('Writing dags to: {}'.format(args.output_dags_dir))
    for i, dag in enumerate(create_dags(dataset, botometer_features=args.botometer)):
        dag_path = os.path.join(args.output_dags_dir, 'dag-{}.json'.format(i))
        write_json(dag, dag_path)


if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s', 
        level=logging.INFO)

    parser = argparse.ArgumentParser(
        epilog="Example: python tweets_to_dags.py"
    )
    parser.add_argument(
        "--input-users-dir",
        help="Input directory containing user profiles as json files",
        dest="input_users_dir",
        type=str,
        default="raw_data/user_profiles"
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
        "--input-botometer-dir",
        help="Input directory containing botometer statistics per user as json files",
        dest="input_botometer_dir",
        type=str,
        default="raw_data/botometer"
    )
    parser.add_argument(
        "--output-dags-dir",
        help="Output directory to exports the dags",
        dest="output_dags_dir",
        type=str,
        default="raw_data/dags"
    )
    parser.add_argument('--botometer', dest='botometer', action='store_true', help="Enable botometer (default is disabled)")
    parser.add_argument('--no-botometer', dest='botometer', action='store_false', help="Disable botometer")
    parser.set_defaults(botometer=False)

    args = parser.parse_args()
    run(args)
