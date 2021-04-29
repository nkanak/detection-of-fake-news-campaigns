#!/usr/bin/env python

import logging
import argparse
import jgrapht
from jgrapht.utils import IntegerSupplier
import numpy as np
import json
import utils
import os

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import f1_score
from tqdm import tqdm


def dag_to_data(filename, ignore_attributes={"ID"}, embeddings_dimensions=200):
    logging.debug("Reading {}".format(filename))

    # read label
    with open(filename) as json_file:
        data = json.load(json_file)
        label = data["label"]
        # 0 = true, 1 = fake
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
            as_list.extend([0] * embeddings_dimensions)
        vfeatures.append(as_list)

    vlabels = []
    vlabels.append(is_fake)

    edge_sources = []
    edge_targets = []
    for e in g.edges:
        edge_sources.append(g.edge_source(e))
        edge_targets.append(g.edge_target(e))

    x = torch.tensor(vfeatures, dtype=torch.float)
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    y = torch.tensor(vlabels, dtype=torch.long)
    result = Data(x=x, edge_index=edge_index, y=y)

    return result


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = GATConv(num_features, 32, heads=4)
        self.conv2 = GATConv(32 * 4, num_classes, heads=1, concat=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=-1)


def train(model, loader, device, optimizer, loss_op):
    model.train()

    total_loss = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        y_true = data.y
        loss = loss_op(y_pred, y_true)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.to(device))
        preds.append(torch.argmax(out, dim=1).cpu())

    y = torch.cat(ys, dim=0).numpy()
    pred = torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


def run(args):

    logging.info("Loading dags dataset")

    data_list = []
    for fentry in os.scandir(args.input_dir):
        if fentry.path.endswith(".json") and fentry.is_file():
            data = dag_to_data(fentry.path)            
            data_list.append(data)
    dataset_size = len(data_list)
    logging.info("Loaded {} dags".format(dataset_size))

    train_size= int(0.8 * dataset_size)
    train_dataset = data_list[:train_size]
    val_dataset = data_list[train_size:]

    # TODO: add test dataset

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_features=208, num_classes=2).to(device)
    loss_op = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)    

    # Start training
    for epoch in range(1, 101):
        logging.info("Starting epoch {}".format(epoch))
        loss = train(model, train_loader, device, optimizer, loss_op)
        val_f1 = test(model, val_loader, device)
        print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}'.format(epoch, loss, val_f1))        

    # TODO: perform test
    #test_f1 = test(model, test_loader, device)
    #print('Val: {:.4f}'.format(test_f1))
    


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
