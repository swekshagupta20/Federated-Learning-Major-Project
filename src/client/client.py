import sys
import os
import json
import torch
import flwr as fl
from collections import OrderedDict
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset

# 1. Path Setup: Ensure the root directory is in sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now internal imports will work perfectly
from src.models.cnn import Net, train, test
from src.utils.hardware_sim import HardwareManager

# Path to the shared metadata
METADATA_PATH = os.path.join(project_root, 'data', 'partition_metadata.json')

def load_data(client_id: int):
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    # Extract indices for this client
    client_indices = metadata[str(client_id)]['indices']
    train_shard = Subset(trainset, client_indices)
    
    return DataLoader(train_shard, batch_size=32, shuffle=True), DataLoader(testset, batch_size=32)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.hardware = HardwareManager() # Phase 2: Sensor link
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
        
        # --- Phase 2: Hardware Reporting ---
        stats = self.hardware.get_sensor_data(self.client_id)
        print(f"\n[CLIENT {self.client_id}] Status -> Battery: {stats[0]}% | Latency: {stats[1]}ms")
        
        train(self.net, self.trainloader, epochs=1)
        
        # --- Phase 2: Battery Consumption ---
        self.hardware.simulate_drain(self.client_id, energy_cost=5.0)
        print(f"[CLIENT {self.client_id}] Training complete. Energy consumed.")
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "battery": stats[0],
            "latency": stats[1]
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    import sys
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient(client_id=cid).to_client())