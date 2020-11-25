#!/usr/bin/env python

import argparse
import json
import time
import os
import random

from datetime import datetime, timedelta

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


def build_initial_graph():
    """Read tweets from a json directory and create a first basic
    graph where tweets are vertices and edges correspond to retweets.
    """
    print("Creating initial graph from tweets")

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

                v = g.add_vertex(tweet["id"])
                g.vertex_attrs[v].update(**tweet)

                if retweet is not None:
                    u = g.add_vertex(retweet["id"])
                    g.vertex_attrs[u].update(**retweet)

                    g.add_edge(v, u)

    print("Created graph with {} vertices".format(g.number_of_vertices))
    print("Created graph with {} edges".format(g.number_of_edges))

    for v in g.vertices:
        created_at_str = g.vertex_attrs[v]["created_at"]
        created_at_obj = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
        g.vertex_attrs[v]["created_at"] = created_at_obj

    return g


def build_dag(
    g,
    v,
    min_retweets=5,
    bucket_delta=timedelta(hours=5),
    tweet_delta=timedelta(minutes=10),
):
    """Build a single retweet dag."""
    if g.outdegree_of(v) != 0:
        raise ValueError("First tweet must not be a retweet")

    if g.indegree_of(v) < min_retweets:
        return None

    print("Building dag starting from {}-{}".format(v, g.vertex_attrs[v]))

    retweets = [g.opposite(e, v) for e in g.inedges_of(v)]
    retweets = sorted(
        retweets, key=lambda x: g.vertex_attrs[v]["created_at"], reverse=True
    )

    chains = []
    while len(retweets) != 0:
        chain = []
        chain_first_tweet = retweets.pop()
        chain_last_tweet = chain_first_tweet
        chain.append(chain_first_tweet)
        chain_datetime_start = g.vertex_attrs[chain_first_tweet]["created_at"]
        chain_datetime_end = chain_datetime_start

        while len(retweets) != 0:
            candidate = retweets[-1]
            candidate_datetime = g.vertex_attrs[candidate]["created_at"]
            if candidate_datetime >= chain_datetime_start + bucket_delta:
                break
            chain.append(retweets.pop())
            chain_last_tweet = candidate
            chain_datetime_end = candidate_datetime

        while len(retweets) != 0:
            candidate = retweets[-1]
            candidate_datetime = g.vertex_attrs[candidate]["created_at"]
            if candidate_datetime >= chain_datetime_end + tweet_delta:
                break
            chain.append(retweets.pop())
            chain_last_tweet = candidate
            chain_datetime_end = candidate_datetime

        chains.append(chain)

    dag = jgrapht.create_graph(directed=True, any_hashable=True)

    g_to_dag = {}
    prev_chain_g_v = None
    for chain in chains:
        prev_g_v = None
        for g_v in chain:
            dag_v = dag.add_vertex()
            g_to_dag[g_v] = dag_v

            dag.vertex_attrs[dag_v]["text"] = g.vertex_attrs[g_v]["text"]

            if prev_g_v is not None:
                dag.add_edge(dag_v, g_to_dag[prev_g_v])
                dag.vertex_attrs[dag_v]["delay"] = abs(
                    (
                        g.vertex_attrs[g_v]["created_at"]
                        - g.vertex_attrs[prev_g_v]["created_at"]
                    ).total_seconds()
                )
            else:
                dag.vertex_attrs[dag_v]["delay"] = 0
            prev_g_v = g_v

        if prev_chain_g_v is not None:
            # connect last vertex of chain with random from last chain
            dag.add_edge(g_to_dag[prev_g_v], g_to_dag[prev_chain_g_v])
            pass

        prev_chain_g_v = random.choice(chain)

    return dag


def build_dags(g):
    """Build a set of dags. Starting from vertices
    which are original tweets, build chains of retweets.
    """
    dags = []

    while True:
        if g.number_of_vertices == 0:
            break

        progress = False
        for v in g.vertices:
            if g.outdegree_of(v) != 0:
                continue

            # isolated
            if g.indegree_of(v) == 0:
                progress = True
                g.remove_vertex(v)
                break

            # start of chain
            d = build_dag(g, v)
            if d is not None:
                dags.append(d)
            progress = True
            g.remove_vertex(v)
            break

        if not progress:
            raise ValueError("All tweets have outgoing edges!")

    return dags


def run(args):

    g = build_initial_graph()
    write_json(g, args.output_file)
    print("Wrote graph as json file: {}".format(args.output_file))

    dags = build_dags(g)
    print("Created dags")

    for dag in dags:
        print(dag)
        print(dag.vertex_attrs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        epilog="Example: python tweets_to_graph.py --input-dir raw_data --output-file tweets_graph.json"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing tweets as json files",
        dest="input_dir",
        type=str,
        default="raw_data",
    )
    parser.add_argument(
        "--output-file",
        help="Output filename to export the graph",
        dest="output_file",
        type=str,
        default="tweets_graph.json",
    )
    args = parser.parse_args()
    run(args)
