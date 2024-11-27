import numpy as np
import torch
import torch.nn.functional as F

class GraphNodeFeaturesExtraction:
    def __init__(self, input_dim, num_layers=2):
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        """ Initialize weight matrices for each layer of GCN """
        return [np.random.rand(self.input_dim, self.input_dim) for _ in range(self.num_layers)]

    def graph_convolution(self, adjacency_matrix, node_features):
        """ Perform a single graph convolution operation. """
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        normalized_adjacency = np.linalg.inv(degree_matrix) @ adjacency_matrix
        normalized_adjacency = normalized_adjacency + np.eye(adjacency_matrix.shape[0])  # Self-connections
        return normalized_adjacency @ node_features

    def extract_features(self, adjacency_matrix, node_features):
        """ Extract node features using multiple layers of graph convolution. """
        features = node_features
        for layer in range(self.num_layers):
            features = self.graph_convolution(adjacency_matrix, features)
            features = self.activation_function(features)
        return features

    def activation_function(self, x):
        """ Apply ReLU activation function. """
        return np.maximum(0, x)


class NodeEnsembleTransferLearning:
    def __init__(self, models):
        self.models = models

    def ensemble_predictions(self, features):
        predictions = np.zeros((features.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(features)

        return self.aggregate(predictions)

    def aggregate(self, predictions):
        aggregated = np.mean(predictions, axis=1)
        return np.argmax(aggregated, axis=1)


class GraphWeightEnsemble:
    def __init__(self, num_nodes, num_classes):
        self.num_nodes = num_nodes
        self.num_classes = num_classes

    def apply_weights(self, node_features, weights):
        """ Apply the aggregate weights to the node features. """
        return node_features * weights


class Classifier:
    def __init__(self, input_dim, num_classes, hidden_units=64):
        self.weights_input_hidden = np.random.rand(input_dim, hidden_units)
        self.weights_hidden_output = np.random.rand(hidden_units, num_classes)
        self.bias_hidden = np.zeros(hidden_units)
        self.bias_output = np.zeros(num_classes)

    def forward(self, x):
        hidden_layer = self.activation_function(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        output_layer = np.dot(hidden_layer, self.weights_hidden_output) + self.bias_output
        return self.softmax(output_layer)

    def activation_function(self, z):
        return np.maximum(0, z)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(self, x):
        class_scores = self.forward(x)
        return np.argmax(class_scores, axis=1)


class GATEGCN:
    def __init__(self, input_dim, num_classes, num_nodes, num_models, hidden_units=64):
        self.gnfe = GraphNodeFeaturesExtraction(input_dim)
        self.netl = NodeEnsembleTransferLearning([Classifier(input_dim, num_classes, hidden_units) for _ in range(num_models)])
        self.gwe = GraphWeightEnsemble(num_nodes, num_classes)
        self.classifier = Classifier(input_dim, num_classes, hidden_units)

    def fit(self, adjacency_matrix, node_features, labels):
        # Step 1: Extract features from nodes
        extracted_features = self.gnfe.extract_features(adjacency_matrix, node_features)

        # Step 2: Perform ensemble learning
        predictions = self.netl.ensemble_predictions(extracted_features)

        # Step 3: Apply weights to the node features for classification
        weighted_features = self.gwe.apply_weights(extracted_features, predictions)

        # Step 4: Train the classifier on the weighted features
        self.classifier.train(weighted_features, labels)

    def predict(self, adjacency_matrix, node_features):
        extracted_features = self.gnfe.extract_features(adjacency_matrix, node_features)
        predictions = self.netl.ensemble_predictions(extracted_features)
        return predictions