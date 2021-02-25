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

from jgrapht.io.exporters import write_json

from dataset import FakeNewsDataset
from dags import create_dags


def run(args):

    logging.info("Loading dataset")

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


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        epilog="Example: python fakenews_tweets_to_dags.py"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing the fakenewsnet dataset",
        dest="input_dir",
        type=str, 
        required=True
    )
    parser.add_argument(
        "--website",
        help="Either politifact or gossipcop",
        dest="website",
        type=str,
        default="politifact",
    )
    parser.add_argument(
        "--output-dags-dir",
        help="Output directory to exports the dags",
        dest="output_dags_dir",
        type=str,
        default="dags",
    )
    args = parser.parse_args()
    run(args)
