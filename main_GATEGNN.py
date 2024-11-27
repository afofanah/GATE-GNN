import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, auc, roc_curve
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import NormalizeFeatures
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# Function to parse command-line arguments for hyperparameters
def parse_args():
    parser = argparse.ArgumentParser(description='Graph Neural Network Training')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--model', type=str, choices=['GATEGNN', 'GATEGCN', 'GATESAGE'], required=True, help='Select model to train')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    return args

# Parse arguments
args = parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Load the dataset
dataset = Planetoid(root='data/Planetoid', name=args.dataset, transform=NormalizeFeatures())
data = dataset[0]

# Create DataLoader for the entire dataset
loader = DataLoader([data], batch_size=1, shuffle=True)

# Select and initialize the model based on the argument
if args.model == 'GATEGNN':
    model = GATEGNN(input_dim=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=args.hidden_dim)
elif args.model == 'GATEGCN':
    model = GATEGCN(input_dim=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=args.hidden_dim)
elif args.model == 'GATESAGE':
    model = GATESAGE(input_dim=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=args.hidden_dim)

# Define Loss Function and Optimizer
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

# Early stopping parameters
best_val_accuracy = 0.0
counter = 0

# Create indices for train, validation, and test splits
num_nodes = data.num_nodes
train_val_indices, test_indices = train_test_split(range(num_nodes), test_size=0.2, random_state=args.seed)
train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, random_state=args.seed)

# Initialize masks as tensors of zeros
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Set the indices in the masks to True
data.train_mask[train_indices] = True
data.val_mask[val_indices] = True
data.test_mask[test_indices] = True

# Lists to store metrics for visualization
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
auc_scores = []

# Training and Evaluation loop
for epoch in range(1, args.epochs + 1):
    # Train the model
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train_samples = 0

    for batch in loader:
        optimizer.zero_grad()
        output = model.fit(batch.edge_index, batch.x, batch.y)  # Use model's fit method
        loss = F.cross_entropy(output[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Calculate train accuracy
        pred_train = output[batch.train_mask].argmax(dim=1)
        correct_train += pred_train.eq(batch.y[batch.train_mask]).sum().item()
        total_train_samples += batch.y[batch.train_mask].size(0)

    # Average the loss and accuracy
    train_loss = total_train_loss / len(loader.dataset)
    train_losses.append(train_loss)
    train_accuracy = correct_train / total_train_samples * 100
    train_accuracies.append(train_accuracy)

    # Evaluate the model
    model.eval()
    total_test_loss = 0
    correct_test = 0
    total_test_samples = 0
    y_true = []
    y_score = []

    with torch.no_grad():
        # Validate
        val_logits = model.predict(data.edge_index, data.x)[data.val_mask]
        val_pred_labels = val_logits.argmax(dim=1)
        val_accuracy = accuracy_score(data.y[data.val_mask].cpu(), val_pred_labels.cpu())

        for batch in loader:
            output = model.predict(batch.edge_index, batch.x)

            # Calculate test loss
            total_test_loss += F.cross_entropy(output[data.test_mask], data.y[data.test_mask]).item()

            # Calculate test accuracy
            pred_test = output[data.test_mask].argmax(dim=1)
            correct_test += pred_test.eq(data.y[data.test_mask]).sum().item()
            total_test_samples += data.y[data.test_mask].size(0)

            # For AUC calculation
            y_true.extend(data.y[data.test_mask].cpu().numpy())
            y_score.extend(F.softmax(output[data.test_mask], dim=1).cpu().numpy())

    test_loss = total_test_loss / len(loader.dataset)
    test_losses.append(test_loss)
    test_accuracy = correct_test / total_test_samples * 100
    test_accuracies.append(test_accuracy)

    # Calculate AUC score
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    auc_macro = roc_auc_score(y_true, y_score, multi_class='ovr') if len(set(y_true)) > 1 else 0.0
    auc_scores.append(auc_macro)

    # Print metrics
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
          f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, AUC Macro: {auc_macro:.4f}')

    # Learning rate scheduler step
    scheduler.step(test_accuracy)

    # Early stopping logic
    if test_accuracy > best_val_accuracy:
        best_val_accuracy = test_accuracy
        counter = 0
    else:
        counter += 1
        if counter >= args.patience:
            print("Early stopping. No improvement in validation accuracy.")
            break

# Plotting the training and test losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plotting training and test accuracies
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Train and Test Accuracies')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and plot the confusion matrix
conf_matrix = confusion_matrix(y_true, np.argmax(y_score, axis=1))

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true),
            yticklabels=np.unique(y_true))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# Calculate F1 scores
f1_score_macro = f1_score(y_true, np.argmax(y_score, axis=1), average='macro')
f1_score_micro = f1_score(y_true, np.argmax(y_score, axis=1), average='micro')
f1_score_weighted = f1_score(y_true, np.argmax(y_score, axis=1), average='weighted')
# Print final evaluation metrics
print(f'Final Test Accuracy: {test_accuracy:.4f}')
print(f'Final AUC Score: {auc_macro:.4f}')
print(f'Test F1-score (Macro): {f1_score_macro:.4f}')
print(f'Test F1-score (Micro): {f1_score_micro:.4f}')
print(f'Test F1-score (Weighted): {f1_score_weighted:.4f}')

# Optional: Embedding visualization using t-SNE
embeddings = output[data.test_mask].detach().cpu().numpy()
from sklearn.manifold import TSNE

# Perform t-SNE to reduce dimensions to 2D for visualization
tsne = TSNE(n_components=2, random_state=args.seed)
embeddings_2d = tsne.fit_transform(embeddings)

# Plotting the t-SNE visualization
plt.figure(figsize=(15, 10))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_true, cmap='viridis', alpha=0.7)
plt.colorbar(label='Node Label')

# Creating a legend
handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
legend_labels = [f'Class {i}' for i in np.unique(y_true)]
plt.legend(handles, legend_labels, title="Classes", loc='best')

plt.title('Node Embedding Visualization with t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(False)
plt.show()
