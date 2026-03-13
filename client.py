import flwr as fl
import torch
from collections import OrderedDict
from model import Net, train, test
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

# 1. Load the Model
net = Net()

# 2. Load Data (Simulating one client's data)
def load_data(partition_id: int = 0, num_partitions: int = 10):
    """
    Load CIFAR-10 data. 
    partition_id: The specific index for this client.
    num_partitions: Total number of clients in the simulation.
    """
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Download the full datasets
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)

    # PLACEHOLDER FOR PARTNER'S LOGIC:
    # This is where we will apply the Dirichlet indices later.
    # For now, it just splits the data evenly (IID).
    partition_size = len(trainset) // num_partitions
    lengths = [partition_size] * num_partitions
    datasets = torch.utils.data.random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))

    # Select the specific partition for this client
    ds = datasets[partition_id]
    
    return DataLoader(ds, batch_size=32, shuffle=True), DataLoader(testset)

trainloader, testloader = load_data()

# 3. Define the Flower Client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# 4. Start Client
if __name__ == "__main__":
    fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient().to_client())