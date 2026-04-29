import os
import sys
import json
import torch
import flwr as fl
from collections import OrderedDict
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------

from src.models.cnn import Net, train, test
from src.simulation.hardware_sim import HardwareManager

METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "partition_metadata.json")


# ---------------------------------------------------
# LOAD DATA (Non-IID)
# ---------------------------------------------------

def load_data(client_id: int):
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CIFAR10("./data", train=True, download=True, transform=transform)
    testset = CIFAR10("./data", train=False, download=True, transform=transform)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    indices = metadata[str(client_id)]["indices"]
    train_subset = Subset(trainset, indices)

    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)

    return trainloader, testloader


# ---------------------------------------------------
# FLOWER CLIENT
# ---------------------------------------------------

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.device = "cpu"

        self.net = Net()
        self.trainloader, self.testloader = load_data(client_id)

        self.hardware = HardwareManager()

    # ------------------------------
    # PARAMETERS
    # ------------------------------

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    # ------------------------------
    # TRAINING
    # ------------------------------

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # 🔥 Simulate network changes
        self.hardware.simulate_network(self.client_id)

        # 🔥 Check dropout (VERY IMPORTANT)
        if not self.hardware.is_available(self.client_id):
            print(f"[CLIENT {self.client_id}] ❌ DROPPED OUT")
            return [], 0, {}

        # Get hardware state
        battery, latency, reliability = self.hardware.get_sensor_data(self.client_id)

        print(f"\n[CLIENT {self.client_id}]")
        print(f"Battery: {battery:.2f}% | Latency: {latency:.2f} ms | Reliability: {reliability:.2f}")

        # Train
        train(self.net, self.trainloader, epochs=1, device=self.device)

        # Simulate battery drain
        self.hardware.simulate_drain(self.client_id, energy_cost=5.0)

        # Return update
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "battery": battery,
            "latency": latency,
            "reliability": reliability
        }

    # ------------------------------
    # EVALUATION
    # ------------------------------

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, accuracy = test(self.net, self.testloader, device=self.device)

        return loss, len(self.testloader.dataset), {
            "accuracy": accuracy
        }


# ---------------------------------------------------
# START CLIENT
# ---------------------------------------------------

if __name__ == "__main__":
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print(f"[STARTING CLIENT {cid}]")

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_id=cid).to_client(),
    )