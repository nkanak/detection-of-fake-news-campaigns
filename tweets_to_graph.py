#!/usr/bin/env python

import argparse
import json
import time
import os
import jgrapht


ATTRS = ['created_at', 'text', 'lang', 'retweet_count' ]

def strip_tweet(dict_tweet):
    tweet = {}
    tweet["id"] = dict_tweet["id_str"]

    for attr in ATTRS: 
        tweet[attr] = dict_tweet[attr]

    return tweet


def run(args):

    g = jgrapht.create_graph(
        directed=True,
        allowing_self_loops=True,
        allowing_multiple_edges=True,
        any_hashable=True,
    )

    for fentry in os.scandir(args.input_dir):
        if fentry.path.endswith(".json") and fentry.is_file():
            with open(fentry.path) as json_file:
                full_tweet = json.load(json_file)
                tweet = strip_tweet(full_tweet)

                v = g.add_vertex(tweet['id'])
                for attr in ATTRS: 
                    g.vertex_attrs[v][attr] = tweet[attr]

            # add edges


    print ('Created graph with {} vertices'.format(g.number_of_vertices))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        epilog="Example: python tweets_to_graph.py --input-dir"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing tweets as json files",
        dest="input_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    run(args)
