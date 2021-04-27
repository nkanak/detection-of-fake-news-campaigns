#!/usr/bin/env python

import logging
import argparse
import jgrapht
from jgrapht.utils import IntegerSupplier
import numpy as np
import torch
import json
import utils

from torch_geometric.data import Dataset, Data


def dag_to_data(filename, ignore_attributes={"ID"}, embeddings_dimensions=200):
    logging.info("Reading {}".format(filename))

    # read label 
    with open(filename) as json_file:
        data = json.load(json_file)
        label = data["label"]
        is_fake = label == "fake"

    # read graph
    g = jgrapht.create_graph()
    vattrs = {}

    def vattrs_cb(v, key, value):
        if v not in vattrs:
            vattrs[v] = {}
        vattrs[v][key] = value

    jgrapht.io.importers.read_json(g, filename, vertex_attribute_cb=vattrs_cb)

    vfeatures = []
    for v in g.vertices:
        as_list = []

        for key in [
            "delay",
            "protected",
            "following_count",
            "listed_count",
            "statuses_count",
            "followers_count",
            "favourites_count",
            "verified",
        ]:
            as_list.append(float(vattrs[v][key]))

        if "user_profile_embedding" in vattrs[v]: 
            as_list.extend(json.loads(vattrs[v]["user_profile_embedding"]))
        else:
            as_list.extend([0]*embeddings_dimensions)
        vfeatures.append(as_list)

    edge_sources = []
    edge_targets = []
    for e in g.edges:
        edge_sources.append(g.edge_source(e))
        edge_targets.append(g.edge_target(e))

    x = torch.tensor(vfeatures, dtype=torch.float)
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    y = torch.tensor([is_fake], dtype=torch.bool)
    result = Data(x=x, edge_index=edge_index, y=y)

    return result


def run(args):

    filename = "{}/dag-1.json".format(args.input_dir)
    data = dag_to_data(filename)
    device = torch.device("cuda")
    data = data.to(device)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python train_dags.py")
    parser.add_argument(
        "--input-dir",
        help="Input directory containing the fakenewsnet dataset",
        dest="input_dir",
        type=str,
        required=True,
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
