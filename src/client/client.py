import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# NOW your imports will work:
from src.models.cnn import Net, train, test

import flwr as fl
import torch
import json
import os
from collections import OrderedDict
from src.models.cnn import Net, train, test # this line changed when "module not found error occured"
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset

# Define path to metadata relative to root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'partition_metadata.json')

def load_data(client_id: int):
    """
    Loads specific shard for a given client_id using the metadata JSON.
    """
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load the global dataset
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)

    # Load your partition map
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    # Extract indices for this specific client
    client_indices = metadata[str(client_id)]['indices']
    
    # Create the specific shard using Subset
    train_shard = Subset(trainset, client_indices)
    
    return DataLoader(train_shard, batch_size=32, shuffle=True), DataLoader(testset, batch_size=32)

# --- Updated Client Class ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.trainloader, self.testloader = load_data(client_id)
        self.net = Net()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    # In a real scenario, pass the client_id as a command-line argument
    import sys
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient(client_id=cid).to_client())