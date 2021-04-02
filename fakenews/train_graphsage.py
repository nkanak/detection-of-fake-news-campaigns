#!/usr/bin/env python

#
# See https://stellargraph.readthedocs.io/en/latest/demos/node-classification/graphsage-inductive-node-classification.html
#

import argparse
import pandas as pd

from stellargraph import StellarDiGraph
import os

from tqdm import tqdm
import json

from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE

from tensorflow.keras import layers, losses, Model
from sklearn import preprocessing, model_selection
import utils

import random

def create_user_labels_df(user_ids, args):
    labels = []
    indexes = []
    count1 = 0
    count2 = 0
    for user_id in tqdm(user_ids):
        if not os.path.exists('%s/%s.json' % (args.user_labels_dir, user_id)):
            count1 += 1
            indexes.append(int(user_id))
            labels.append(random.randint(0, 1))
            continue

        with open('%s/%s.json' % (args.user_labels_dir, user_id)) as json_file:
            user_label = json.load(json_file)
            indexes.append(int(user_label['id']))
            if user_label['fake'] >= user_label['real']:
                labels.append(1)
            else:
                labels.append(0)
            count2 += 1
    df = pd.DataFrame(labels, index=indexes, columns=['label'])
    print('We set random label to %s users' % (count1))
    print('We set correct labels to %s users' % (count2))
    return df


def run(args):
    edges_df =  utils.read_pickle_from_file('edges.pkl')
    vertices_df = utils.read_pickle_from_file('vertices.pkl')
    vertices_df.drop(['id'], inplace=True, axis=1)
    labels = create_user_labels_df(list(vertices_df.index), args)
    labels = labels['label']
    labels_sampled = labels.sample(frac=0.8, replace=False, random_state=101)

    print('Create StellarGraph graph')
    g = StellarDiGraph(vertices_df, edges_df, edge_type_default='follows', node_type_default='user')
    print(g.info())
    graph_sampled = g.subgraph(list(labels_sampled.index))

    train_labels, test_labels = model_selection.train_test_split(
        labels_sampled,
        train_size=0.2,
        test_size=None,
        stratify=labels_sampled,
        random_state=42,
    )
    val_labels, test_labels = model_selection.train_test_split(
        test_labels, train_size=0.2, test_size=None, stratify=test_labels, random_state=100,
    )

    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_labels)
    val_targets = target_encoding.transform(val_labels)
    test_targets = target_encoding.transform(test_labels)

    batch_size = 100
    num_samples = [25, 10]

    generator = GraphSAGENodeGenerator(graph_sampled, batch_size, num_samples)
    train_gen = generator.flow(list(train_labels.index), train_targets, shuffle=True)

    graphsage_model = GraphSAGE(
        layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.5,
    )
    x_inp, x_out = graphsage_model.in_out_tensors()
    prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
    print(prediction.shape)

    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer='adam',
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )

    val_gen = generator.flow(list(val_labels.index), val_targets)

    history = model.fit(
        train_gen, epochs=15, validation_data=val_gen, verbose=1, shuffle=False
    )

    test_gen = generator.flow(test_labels.index, test_targets)
    test_metrics = model.evaluate(test_gen)
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="Example: python train_gcn.py --input-filename tweets_graph.json"
    )
    parser.add_argument(
        "--user-labels-dir",
        help="Directory of user labels",
        dest="user_labels_dir",
        type=str,
        required=True
    )
    args = parser.parse_args()
    run(args)