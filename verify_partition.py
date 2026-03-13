import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt

import json
import numpy as np
import seaborn as sns
from torchvision import datasets

def verify_partition(metadata_path='partition_metadata.json'):
    # Load metadata
    with open(metadata_path, 'r') as f:
        partition = json.load(f)
    
    # Load dataset to get class labels
    train_data = datasets.CIFAR10(root='./data', train=True, download=True)
    targets = np.array(train_data.targets)
    
    # Create matrix: [num_clients x num_classes]
    num_clients = len(partition)
    num_classes = 10
    distribution_matrix = np.zeros((num_clients, num_classes))
    
    for client_id, indices in partition.items():
        client_labels = targets[indices]
        for label in range(num_classes):
            distribution_matrix[int(client_id), label] = np.sum(client_labels == label)
            
    # Visualize
    plt.figure(figsize=(12, 8))
    sns.heatmap(distribution_matrix, cmap="YlGnBu", cbar=True)
    plt.title(f"Non-IID Data Distribution (Alpha=0.5)")
    plt.xlabel("CIFAR-10 Classes")
    plt.ylabel("Client ID")
    plt.tight_layout()

    #Saving the heatmap instead of showing it 
    plt.savefig('distribution_heatmap.png')
    print("Heatmap saved as 'distribution_heatmap.png'")

if __name__ == "__main__":
    verify_partition()