import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class GraphNodeFeaturesExtraction(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super(GraphNodeFeaturesExtraction, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim

    def sample_neighbors(self, adjacency_matrix, node_idx, num_samples):
        """ Sample neighbors for a given node. """
        neighbors = np.where(adjacency_matrix[node_idx] > 0)[0]
        if len(neighbors) > num_samples:
            return np.random.choice(neighbors, num_samples, replace=False)
        return neighbors

    def aggregate(self, adjacency_matrix, node_features, node_idx, num_samples):
        """ Aggregate features from sampled neighbors and the node itself. """
        neighbors = self.sample_neighbors(adjacency_matrix, node_idx, num_samples)
        neighbor_features = node_features[neighbors]

        # Simple aggregation: Mean
        aggregated_features = np.mean(neighbor_features, axis=0)
        return np.concatenate((node_features[node_idx], aggregated_features))

    def extract_features(self, adjacency_matrix, node_features):
        """ Extract features using GraphSAGE mechanisms. """
        features = node_features
        for layer in range(self.num_layers):
            new_features = np.zeros_like(features)
            for node in range(features.shape[0]):
                new_features[node] = self.aggregate(adjacency_matrix, features, node, num_samples=2)  # Example: sample 2 neighbors
            features = new_features
        return features


class NodeEnsembleTransferLearning(nn.Module):
    def __init__(self, models):
        super(NodeEnsembleTransferLearning, self).__init__()
        self.models = models

    def ensemble_predictions(self, features):
        predictions = np.zeros((features.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(features)

        return self.aggregate(predictions)

    def aggregate(self, predictions):
        aggregated = np.mean(predictions, axis=1)
        return np.argmax(aggregated, axis=1)


class GraphWeightEnsemble(nn.Module):
    def __init__(self, num_nodes, num_classes):
        super(GraphWeightEnsemble, self).__init__()
        self.num_nodes = num_nodes
        self.num_classes = num_classes

    def apply_weights(self, node_features, weights):
        """ Apply weights to the node features. """
        return node_features * weights


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_units=64):
        super(Classifier, self).__init__()
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


class GATESAGE(nn.Module):
    def __init__(self, input_dim, num_classes, num_nodes, num_models, hidden_units=64):
        super(GATESAGE, self).__init__()
        self.gnfe = GraphNodeFeaturesExtraction(input_dim)
        self.netl = NodeEnsembleTransferLearning([Classifier(input_dim, num_classes, hidden_units) for _ in range(num_models)])
        self.gwe = GraphWeightEnsemble(num_nodes, num_classes)
        self.classifier = Classifier(input_dim, num_classes, hidden_units)

    def fit(self, adjacency_matrix, node_features, labels):
        # Step 1: Extract features from nodes using GraphSAGE
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