from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from absl import flags
from absl.flags import FLAGS
from enum import Flag
import time
import pydot
import graphviz
import os
import sys
from matplotlib import pyplot as plt
from inits import *
from scipy.sparse.linalg import eigsh
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures, num_classes
from model import MLP
from model import AdversarialAttack
from model import GATE_GNN
from model import TransferLearning
import argparse


# Settings

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.') #8, 16, 128
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--early_stopping', type=int, default=100, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', type=int,default=3, help='Maximum Chebyshev polynomial degree.') #3, 5, 7, 11
parser.add_argument('--imbalance_ratio', type=float, default=0.2, help='Imabalance ratio') # 0.2, 0.4,.. 0.6
parser.add_argument('--lamnda', type=float, default=0, help='parameter sensitivity') #
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers') #2, 3, or 4

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load the dataset
dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())

# Split the dataset into train, val, and test
data = dataset[0]
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[:data.num_nodes - 1000] = True
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask[data.num_nodes - 500:data.num_nodes - 250] = True
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[data.num_nodes - 250:] = True

# Define the model, optimizer, and loss function
model = GATE_GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Define a function to compute the accuracy
def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean()

# Train the model
num_epochs = 200
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    # Forward pass
    logits = model(data.x, data.edge_index)
    # Compute the loss
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Compute the train accuracy
    train_acc = accuracy(logits[data.train_mask], data.y[data.train_mask])
    # Set the model to evaluation mode
    model.eval()
    # Compute the validation accuracy
    val_acc = accuracy(logits[data.val_mask], data.y[data.val_mask])
    # Print the results
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Test the model
model.eval()
test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
print(f"Test Acc: {test_acc:.4f}")


# Define the attack model, optimizer, and loss function
attack_model = AdversarialAttack(model, data.num_nodes, data.num_features)
attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01)
attack_criterion = torch.nn.CrossEntropyLoss()

# Define a function to compute the attack success rate
def attack_success_rate(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds != labels).float().mean()

# Attack the model
num_attack_epochs = 100
for epoch in range(num_attack_epochs):
    # Set the attack model to training mode
    attack_model.train()
    # Forward pass
    attack_logits = attack_model(data.x, data.edge_index)
    # Compute the loss
    attack_loss = attack_criterion(attack_logits[data.train_mask], data.y[data.train_mask])
    # Backward pass
    attack_optimizer.zero_grad()
    attack_loss.backward()
    attack_optimizer.step()
    # Compute the attack success rate
    attack_success = attack_success_rate(attack_logits[data.test_mask], data.y[data.test_mask])
    # Print the results
    print(f"Attack Epoch {epoch + 1}, Loss: {attack_loss:.4f}, Success Rate: {attack_success:.4f}")


# Define the transfer model, optimizer, and loss function
transfer_model = TransferLearning(model, num_classes)
transfer_optimizer = torch.optim.Adam(transfer_model.parameters(), lr=0.005, weight_decay=5e-4)
transfer_criterion = torch.nn.CrossEntropyLoss()