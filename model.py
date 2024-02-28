"""
Install geometric
!pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install git+https://github.com/pyg-team/pytorch_geometric.git
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Define the number of features, channels, classes, heads, and layers
num_features = 1433
num_classes = 7
num_heads = 8
num_heads = 8
per_head_channels = 16
num_layers = 2

# Usage based on the dataset: Cora, NELL, Citeseer, and PubMed
num_nodes = 2708
in_features = 64
out_features = 128
num_heads = 8

# For MLP
input_dim = 64
hidden_dim = 128
output_dim = 1
num_layers = 3

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

# GATE-GNN Model here

# Define the GATE-GNN model of MAS class
class MAS(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MAS, self).__init__()
        # Initialize the graph convolution layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer takes the input features
                self.gcn_layers.append(GCNConv(num_features, per_head_channels * num_heads))
            else:
                # Subsequent layers take the output of the previous layer
                self.gcn_layers.append(GCNConv(per_head_channels,  num_heads, per_head_channels,  num_heads))
        # Initialize the attention layers
        self.att_layers = nn.ModuleList()
        for i in range(num_layers):
            # Each layer has a multi-head attention mechanism
            self.att_layers.append(GATConv(per_head_channels * num_heads, per_head_channels, heads=num_heads, concat=False))
        # Initialize the output layer
        self.out_layer = nn.Linear(per_head_channels * num_heads, num_classes)

    def forward(self, x, edge_index):
        # Loop over the layers
        for i in range(num_layers):
            # Apply graph convolution
            x = self.gcn_layersi
            # Apply non-linearity
            x = F.relu(x)
            # Apply attention
            x = self.att_layersi
            # Apply dropout
            x = F.dropout(x, p=0.1, training=self.training)
        # Apply output layer
        x = self.out_layer(x)
        return x

# Define the feature extraction layer class (GNFE)
class GNFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNFE, self).__init__()
        # Initialize a graph convolution layer
        self.gcn = GCNConv(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # Apply graph convolution
        x = self.gcn(x, edge_index)
        # Apply non-linearity
        x = F.relu(x)
        return x
    
# Define the GEWA of the GNN model class
class GEWA(nn.Module):
    def __init__(self):
        super(GEWA, self).__init__()
        # Initialize the feature extraction layer
        self.feature_extraction = GNFE(num_features, num_heads * num_classes)
        # Initialize the attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(num_layers):
            # Each layer has a multi-head attention mechanism
            self.attention_layers.append(MAS(num_heads * num_classes, num_classes))
        # Initialize the output layer
        self.out_layer = nn.Linear(num_heads * num_classes, num_classes)

    def forward(self, x, edge_index):
        # Extract features from the input
        x = self.feature_extraction(x, edge_index)
        # Loop over the layers
        for i in range(num_layers):
            # Apply attention
            x = self.attention_layersi
        # Apply output layer
        x = self.out_layer(x)
        #return x
    def forward(self, x, adjacency_matrix):
        # Compute attention scores for each head
        attention_scores = torch.cat([att(x) for att in self.attentions], dim=1)
        # Apply softmax to get attention coefficients
        attention_coeffs = torch.softmax(attention_scores, dim=1)
        # Combine neighbor features using attention coefficients
        output = torch.matmul(attention_coeffs, x)
        return output
adjacency_matrix = torch.rand(num_nodes, num_nodes)  # Replace with actual adjacency matrix
GEWA_layer = GEWA(in_features, out_features, num_heads)

# Define the adversarial attack class
class AdversarialAttack(nn.Module):
    def __init__(self, model, num_nodes, num_features, epsilon=0.01):
        super(AdversarialAttack, self).__init__()
        # Initialize the model to be attacked
        self.model = model
        # Initialize the number of nodes and features
        self.num_nodes = num_nodes
        self.num_features = num_features
        # Initialize the perturbation vector
        self.perturb = nn.Parameter(torch.FloatTensor(num_nodes, num_features))
        nn.init.uniform_(self.perturb, -epsilon, epsilon)
    
    def forward(self, x, edge_index):
        # Add the perturbation to the input features
        x = x + self.perturb
        # Clip the values to the range [0, 1]
        x = torch.clamp(x, 0, 1)
        # Pass the perturbed input to the model
        out = self.model(x, edge_index)
        return out

# Define the transfer learning class
class TransferLearning(nn.Module):
    def __init__(self, model, num_classes):
        super(TransferLearning, self).__init__()
        # Initialize the model to be transferred
        self.model = model
        # Freeze the parameters of the model
        for param in self.model.parameters():
            param.requires_grad = False
        # Initialize the new output layer
        self.out_layer = nn.Linear(num_heads * num_classes, num_classes)
    
    def forward(self, x, edge_index):
        # Pass the input to the model
        x = self.model(x, edge_index)
        # Pass the output to the new output layer
        x = self.out_layer(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
mlp = MLP(input_dim, hidden_dim, output_dim, num_layers)
output = mlp(in_features)  # Replace 'x' with your input features



# Define the GNN model class
class GATE_GNN(nn.Module):
    def __init__(self):
        super(GATE_GNN, self).__init__()
        # Initialize the feature extraction layer
        self.feature_extraction = GNFE(num_features, num_heads * num_classes)
        # Initialize the attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(num_layers):
            # Each layer has a multi-head attention mechanism
            self.attention_layers.append(MAS(num_heads * num_classes, num_classes))
        # Initialize the ensemble layer
        self.ensemble = GEWA(num_layers, 1)
        # Initialize the output layer
        self.out_layer = nn.Linear(num_heads * num_classes, num_classes)

    def forward(self, x, edge_index):
        # Extract features from the input
        x = self.feature_extraction(x, edge_index)
        # Initialize a list to store the outputs of each layer
        outputs = []
        # Loop over the layers
        for i in range(num_layers):
            # Apply attention
            x = self.attention_layers
            # Append the output to the list
            outputs.append(x)
        # Stack the outputs along the second dimension
          
        outputs = torch.stack(outputs, dim=1)
        # Apply ensemble
        weights = self.ensemble(outputs)
        # Weighted sum of the outputs
        x = torch.sum(weights * outputs, dim=1)
        # Apply output layer
        x = self.out_layer(x)
        return x
    