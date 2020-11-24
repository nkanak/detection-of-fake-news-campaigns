#!/usr/bin/env python

import argparse
import json
import time
import os
import jgrapht
from jgrapht.io.exporters import write_json


TWEET_ATTRS = {
    "id_str": "id",
    "created_at": "created_at",
    "text": "text",
    "lang": "lang",
    "retweet_count": "retweet_count",
}
USER_ATTRS = {"id_str": "userid", "name": "username", "screen_name": "userscreenname"}


def strip_tweet(dict_tweet):
    tweet = {}
    for k, v in TWEET_ATTRS.items():
        if k in dict_tweet:
            tweet[v] = dict_tweet[k]

    for k, v in USER_ATTRS.items():
        if k in dict_tweet["user"]:
            tweet[v] = dict_tweet["user"][k]

    retweet = None
    if "retweeted_status" in dict_tweet:
        retweet, _ = strip_tweet(dict_tweet["retweeted_status"])

    return tweet, retweet


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
                tweet, retweet = strip_tweet(full_tweet)

                v = g.add_vertex(tweet['id'])
                g.vertex_attrs[v].update(**tweet)

                if retweet is not None:
                    u = g.add_vertex(retweet['id'])
                    g.vertex_attrs[u].update(**retweet)

                    g.add_edge(v, u)

    print("Created graph with {} vertices".format(g.number_of_vertices))
    print("Created graph with {} edges".format(g.number_of_edges))

    write_json(g, args.output_file)

    print("Wrote graph as json file: {}".format(args.output_file))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        epilog="Example: python tweets_to_graph.py --input-dir raw_data --output-file tweets_graph.json"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing tweets as json files",
        dest="input_dir",
        type=str,
        default="raw_data"
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
