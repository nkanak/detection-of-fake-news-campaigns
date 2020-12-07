#!/usr/bin/env python

#
# See https://stellargraph.readthedocs.io/en/stable/demos/embeddings/deep-graph-infomax-embeddings.html
#

import argparse
import os
import numbers

import jgrapht
from jgrapht.io.importers import read_json

import pandas as pd

import stellargraph
from stellargraph.mapper import (
    CorruptedGenerator,
    FullBatchNodeGenerator,
)
from stellargraph import StellarGraph
from stellargraph.layer import GCN, DeepGraphInfomax
from stellargraph.utils import plot_history

from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model


def feature_to_numeric(values):
    index = {}
    next_value = 1
    output = []
    for v in values:
        if v not in index:
            index[v] = next_value
            next_value += 1
        output.append(index[v])
    return output


def edges_to_df(g):
    sources = []
    targets = []

    for e in g.edges:
        u = g.edge_source(e)
        v = g.edge_target(e)
        sources.append(u)
        targets.append(v)

    return pd.DataFrame({"source": sources, "target": targets})


def vertices_to_df(g, features):
    data = {}

    vertices = []
    for v in g.vertices:
        vertices.append(v)

    for f in features:
        values = []
        is_numeric = True
        for v in g.vertices:
            v = g.vertex_attrs[v][f]
            if is_numeric and not isinstance(v, numbers.Number):
                is_numeric = False
            values.append(v)

        if not is_numeric:
            values = feature_to_numeric(values)

        data[f] = values

    return pd.DataFrame(data, index=vertices)


def run(args):

    jgrapht_g = jgrapht.create_graph(
        directed=True,
        allowing_self_loops=True,
        allowing_multiple_edges=True,
        any_hashable=True,
    )
    read_json(jgrapht_g, args.input_file)
    print("Read graph with {} vertices".format(jgrapht_g.number_of_vertices))
    print("Read graph with {} edges".format(jgrapht_g.number_of_edges))

    edges_df = edges_to_df(jgrapht_g)
    vertices_df = vertices_to_df(jgrapht_g, features=("created_at", "retweet_count", "userid"))

    g = StellarGraph(vertices_df, edges_df)

    print(g.info())

    fullbatch_generator = FullBatchNodeGenerator(g, sparse=False)
    gcn_model = GCN(layer_sizes=[128, 64], activations=["relu", "relu"], generator=fullbatch_generator)

    corrupted_generator = CorruptedGenerator(fullbatch_generator)
    gen = corrupted_generator.flow(g.nodes())

    infomax = DeepGraphInfomax(gcn_model, corrupted_generator)
    x_in, x_out = infomax.in_out_tensors()

    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))

    epochs = 100
    es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
    history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])
    plot_history(history)

    x_emb_in, x_emb_out = gcn_model.in_out_tensors()

    # for full batch models, squeeze out the batch dim (which is 1)
    x_out = tf.squeeze(x_emb_out, axis=0)
    emb_model = Model(inputs=x_emb_in, outputs=x_out)

    all_embeddings = emb_model.predict(fullbatch_generator.flow(g.nodes()))

    trans = TSNE(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(all_embeddings), index=g.nodes())

    ig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        emb_transformed[0],
        emb_transformed[1],
        cmap="jet",
    )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    plt.title("TSNE visualization of GCN embeddings")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        epilog="Example: python train_gcn.py --input-filename tweets_graph.json"
    )
    parser.add_argument(
        "--input-file",
        help="Input filename to import the graph from json",
        dest="input_file",
        type=str,
        default="tweets_graph.json",
    )
    args = parser.parse_args()
    run(args)
