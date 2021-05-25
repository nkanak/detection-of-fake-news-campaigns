#!/usr/bin/env python

import logging
import argparse
import jgrapht
import json
import os

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm


def tree_to_data(filename):
    logging.debug("Reading {}".format(filename))

    # read label
    with open(filename) as json_file:
        data = json.load(json_file)
        label = data["label"]
        # 0 = true, 1 = fake
        is_fake = label == "fake"

    vfeatures = []
    for node in data['nodes']:
        as_list = []
        for key in ["delay", "protected", "following_count", "listed_count", "statuses_count", "followers_count", "favourites_count", "verified",]:
            as_list.append(float(node[key]))
        as_list.extend(node["embedding"])
        vfeatures.append(as_list)

    vlabels = []
    vlabels.append(is_fake)

    edge_sources = []
    edge_targets = []
    for e in data['edges']:
        edge_sources.append(e['source'])
        edge_targets.append(e['target'])

    x = torch.tensor(vfeatures, dtype=torch.float)
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    y = torch.tensor(vlabels, dtype=torch.long)
    result = Data(x=x, edge_index=edge_index, y=y)

    number_of_features = len(vfeatures[0])
    return number_of_features, result


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
    test_f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    test_precision = precision_score(y, pred) if pred.sum() > 0 else 0
    test_recall = recall_score(y, pred)
    test_accuracy = accuracy_score(y, pred)

    return test_f1, test_precision, test_recall, test_accuracy


def run(args):

    logging.info("Loading dags dataset")

    train_path = os.path.join(args.dataset_root, 'train')
    val_path = os.path.join(args.dataset_root, 'val')
    test_path = os.path.join(args.dataset_root, 'test')

    datasets = []
    for path in [train_path, val_path, test_path]:
        dataset = []
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                number_of_features, t = tree_to_data(fentry.path)
                dataset.append(t)
        datasets.append(dataset)

    train_loader = DataLoader(datasets[0], batch_size=1)
    val_loader = DataLoader(datasets[1], batch_size=4)
    test_loader = DataLoader(datasets[2], batch_size=4)

    logging.info('Train dataset size: %s ' % len(train_loader))
    logging.info('Validation dataset size: %s ' % len(val_loader))
    logging.info('Test dataset size: %s' % len(test_loader))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_features=number_of_features, num_classes=2).to(device)
    loss_op = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)    

    # Start training
    for epoch in range(1, 101):
        logging.info("Starting epoch {}".format(epoch))
        loss = train(model, train_loader, device, optimizer, loss_op)
        val_f1, val_precision, val_recall, val_accuracy = test(model, val_loader, device)
        print('Epoch: {:02d}, Loss: {:.4f}, Val F1: {:.4f} Val Prec: {:.4f} Val Rec: {:.4f} Val Acc: {:.4f}'.format(epoch, loss, val_f1, val_precision, val_recall, val_accuracy))

    test_f1, test_precision, test_recall, test_accuracy = test(model, test_loader, device)
    print('F1: %s Precision: %s Recall: %s Accuracy: %s' % (test_f1, test_precision, test_recall, test_accuracy))
    


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python train_dags.py")
    parser.add_argument(
        "--dataset-root",
        help="Dataset root",
        dest="dataset_root",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    run(args)
