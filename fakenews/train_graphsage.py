#!/usr/bin/env python

#
# See https://stellargraph.readthedocs.io/en/latest/demos/node-classification/graphsage-inductive-node-classification.html
#

import argparse
from stellargraph import StellarGraph
import numpy as np
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from tensorflow.keras import layers, Model, optimizers
from sklearn import preprocessing, model_selection
import utils
import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sklearn
import tensorflow


def run(args):
    edges_df =  utils.read_pickle_from_file('edges.pkl')
    vertices_df = utils.read_pickle_from_file('vertices.pkl')
    vertices_df.drop(['id'], inplace=True, axis=1)
    user_labels_dir = args.user_labels_dir
    labels = utils.create_user_labels_df(list(vertices_df.index), user_labels_dir)['label']
    print('###### Total labels value counts')
    print(labels.value_counts())
    labels_sampled = labels.sample(frac=0.8, replace=False, random_state=100)

    print('###### Labels sampled value counts')
    print(labels_sampled.value_counts())
    train_labels, test_labels = model_selection.train_test_split(
        labels_sampled,
        train_size=None,
        test_size=0.33,
        stratify=labels_sampled,
        random_state=100,
    )
    val_labels, test_labels = model_selection.train_test_split(
        test_labels,
        train_size=0.2,
        test_size=None,
        stratify=test_labels,
        random_state=100,
    )

    print('###### Labels train value counts')
    print(train_labels.value_counts())
    print('###### Labels test value counts')
    print(test_labels.value_counts())
    print('###### Labels val value counts')
    print(val_labels.value_counts())

    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_labels)
    val_targets = target_encoding.transform(val_labels)
    test_targets = target_encoding.transform(test_labels)

    print('###### Create StellarGraph graph')
    graph_full = StellarGraph(vertices_df, edges_df, edge_type_default='follows', node_type_default='user')
    print(graph_full.info())
    graph_sampled = graph_full.subgraph(list(labels_sampled.index))

    batch_size = 100
    num_samples = [20, 10]
    generator = GraphSAGENodeGenerator(graph_sampled, batch_size, num_samples)
    train_gen = generator.flow(list(train_labels.index), train_targets, shuffle=True)

    graphsage_model = GraphSAGE(
        layer_sizes=[32, 20], generator=generator, bias=True, dropout=0.5, activations=['relu', 'relu']
    )
    x_inp, x_out = graphsage_model.in_out_tensors()
    if train_targets.shape[1] == 1:
        print('###### binary classification problem')
        activation = "sigmoid"
        loss = "binary_crossentropy"
        metrics = ["accuracy", tensorflow.keras.metrics.Precision(name='precision'), tensorflow.keras.metrics.Recall(name='recall')]
    else:
        print('###### multi-classification problem')
        activation = "softmax"
        loss = "categorical_crossentropy"
        metrics = ["acc"]
    prediction = layers.Dense(units=train_targets.shape[1], activation=activation)(x_out)
    print('###### prediction shape')
    print(prediction.shape)
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=0.001),
        loss=loss,
        metrics=metrics,
    )
    val_gen = generator.flow(val_labels.index, val_targets)
    history = model.fit(
        train_gen, epochs=2, validation_data=val_gen, verbose=1, shuffle=False
    )

    test_gen = generator.flow(test_labels.index, test_targets)
    test_metrics = model.evaluate(test_gen)
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    #
    # Making predictions with the model
    #
    generator = GraphSAGENodeGenerator(graph_full, batch_size, num_samples)
    hold_out_nodes = labels.index.difference(labels_sampled.index)

    labels_hold_out = labels[hold_out_nodes]
    print('###### labels hold out value counts')
    print(labels_hold_out.value_counts())
    hold_out_targets = target_encoding.transform(labels_hold_out)
    hold_out_gen = generator.flow(hold_out_nodes, hold_out_targets)
    hold_out_predictions = model.predict(hold_out_gen)
    hold_out_predictions = target_encoding.inverse_transform(hold_out_predictions)
    print(len(hold_out_predictions))

    results = pd.Series(hold_out_predictions, index=hold_out_nodes)
    df = pd.DataFrame({"Predicted": results, "True": labels_hold_out})
    print(df.head(10))
    print('###### Precision')
    print(sklearn.metrics.precision_score(labels_hold_out, results))
    print('###### Recall')
    print(sklearn.metrics.recall_score(labels_hold_out, results))
    hold_out_metrics = model.evaluate(hold_out_gen)
    print("\nHold Out Set Metrics:")
    for name, val in zip(model.metrics_names, hold_out_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    #
    # Node embeddings for hold out nodes
    #
    embedding_model = Model(inputs=x_inp, outputs=x_out)
    emb = embedding_model.predict(hold_out_gen)
    print('###### embedding shape')
    print(emb.shape)
    X = emb
    y = np.argmax(target_encoding.transform(labels_hold_out), axis=1)
    if X.shape[1] > 2:
        transform = TSNE  # PCA
        trans = transform(n_components=2)
        emb_transformed = pd.DataFrame(trans.fit_transform(X), index=hold_out_nodes)
        emb_transformed["label"] = y
    else:
        emb_transformed = pd.DataFrame(X, index=hold_out_nodes)
        emb_transformed = emb_transformed.rename(columns={"0": 0, "1": 1})
        emb_transformed["label"] = y

    alpha = 0.7

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        emb_transformed[0],
        emb_transformed[1],
        c=emb_transformed["label"].astype("category"),
        cmap="jet",
        alpha=alpha,
    )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    plt.title(
        "{} visualization of GraphSAGE embeddings of hold out nodes".format(
            transform.__name__
        )
    )
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="Example: python train_graphsage.py --user-labels-dir ../raw_data/user_labels"
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