import numpy as np
import json
import os
from torchvision import datasets, transforms

def partition_data(num_clients=100, alpha=0.5):
    # 1. Load the full dataset (downloading to local path)
    train_data = datasets.CIFAR10(root='./data', train=True, download=True)
    targets = np.array(train_data.targets)
    num_classes = 10
    
    # 2. Get indices per class
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    
    # 3. Create Dirichlet distribution for each class
    client_id_to_indices = {str(i): [] for i in range(num_clients)}
    
    for class_idx in range(num_classes):
        # Dirichlet distribution: how much of this class goes to each client
        proportions = np.random.dirichlet([alpha] * num_clients)
        # Scale proportions to the number of samples in this class
        class_samples = class_indices[class_idx]
        sample_counts = (proportions * len(class_samples)).astype(int)
        
        # Ensure total counts match (fix rounding errors)
        sample_counts[-1] += len(class_samples) - np.sum(sample_counts)
        
        # Distribute indices
        start = 0
        for client_id in range(num_clients):
            end = start + sample_counts[client_id]
            client_id_to_indices[str(client_id)].extend(class_samples[start:end].tolist())
            start = end
            
    # 4. Save to JSON
    with open('partition_metadata.json', 'w') as f:
        json.dump(client_id_to_indices, f, indent=4)
        
    print(f"Successfully partitioned data for {num_clients} clients.")

if __name__ == "__main__":
    partition_data()