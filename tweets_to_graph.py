#!/usr/bin/env python

import argparse
import json
import time
import os
import jgrapht

from jgrapht.io.exporters import write_json

from models import Dataset



def run(args):

    dataset = Dataset()
    dataset.load_users_and_followers(args.input_followers_dir)
    dataset.load_tweets(args.input_tweets_dir)
    #print(dataset.users_by_id)
    #print(dataset.tweets_by_id)
    #print(dataset)


    #write_json(g, args.output_file)
    #print("Wrote graph as json file: {}".format(args.output_file))

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
        "--output-file",
        help="Output filename to export the graph",
        dest="output_file",
        type=str,
        default="tweets_graph.json"
    )
    args = parser.parse_args()
    run(args)
