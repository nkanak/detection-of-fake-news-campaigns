#!/usr/bin/env python

#
# See https://stellargraph.readthedocs.io/en/latest/demos/node-classification/graphsage-inductive-node-classification.html
#

import argparse
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from tensorflow.keras import layers, Model, optimizers
from sklearn import preprocessing, model_selection
import utils
import tensorflow


def infer(embedding_model, vertices_df, edges_df, batch_size, num_samples):
    hold_out_graph = StellarGraph(vertices_df, edges_df, edge_type_default='follows', node_type_default='user')
    hold_out_gen = GraphSAGENodeGenerator(hold_out_graph, batch_size, num_samples)
    hold_out_gen = hold_out_gen.flow(vertices_df.index)
    if type(embedding_model) in [str]:
        embedding_model = tensorflow.keras.models.load_model(embedding_model)
    emb = embedding_model.predict(hold_out_gen)
    return emb

def run(args):
    edges_df =  utils.read_pickle_from_file('edges.pkl')
    vertices_df = utils.read_pickle_from_file('vertices.pkl')
    vertices_df.drop(['id'], inplace=True, axis=1)
    user_labels_dir = args.user_labels_dir
    labels = utils.create_user_labels_df(list(vertices_df.index), user_labels_dir)['label']
    print('###### Total labels value counts')
    print(labels.value_counts())
    labels_sampled = labels.sample(frac=0.8, replace=False, random_state=100)
    vertices_df_sampled = vertices_df.sample(frac=0.8, replace=False, random_state=100)
    edges_df_sampled = edges_df.sample(frac=0.8, replace=False, random_state=100)
    edges_df_sampled = edges_df_sampled[edges_df_sampled['source'].isin(vertices_df_sampled.index.tolist())]
    edges_df_sampled = edges_df_sampled[edges_df_sampled['target'].isin(vertices_df_sampled.index.tolist())]

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
    print(edges_df_sampled)
    print(vertices_df_sampled)
    graph_sampled = StellarGraph(vertices_df_sampled, edges_df_sampled, edge_type_default='follows', node_type_default='user')
    print(graph_sampled.info())

    batch_size = 100
    num_samples = [20, 10]
    generator = GraphSAGENodeGenerator(graph_sampled, batch_size, num_samples)
    train_gen = generator.flow(list(train_labels.index), train_targets, shuffle=True)

    graphsage_model = GraphSAGE(
        layer_sizes=[32, 15], generator=generator, bias=True, dropout=0.5, activations=['relu', 'relu']
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
        train_gen, epochs=1, validation_data=val_gen, verbose=1, shuffle=False
    )

    test_gen = generator.flow(test_labels.index, test_targets)
    test_metrics = model.evaluate(test_gen)
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    hold_out_vertices_df = vertices_df[~vertices_df.index.isin(vertices_df_sampled.index)]
    hold_out_edges_df = edges_df[edges_df['source'].isin(hold_out_vertices_df.index.tolist())]
    hold_out_edges_df = hold_out_edges_df[hold_out_edges_df['target'].isin(hold_out_vertices_df.index.tolist())]

    utils.write_object_to_pickle_file('hold_out_vertices_df.pkl', hold_out_vertices_df)
    utils.write_object_to_pickle_file('hold_out_edges_df.pkl', hold_out_edges_df)
    embedding_model_filename = 'users_embedding_%s_%s_model' % (batch_size, '_'.join([str(i) for i in num_samples]))
    print('Saving embedding_model: %s' % (embedding_model_filename))
    embedding_model = Model(inputs=x_inp, outputs=x_out)
    embedding_model.save(embedding_model_filename)

    print(infer(embedding_model, hold_out_vertices_df, hold_out_edges_df, batch_size, num_samples))

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