import flwr as fl
import torch

# 1. Define the Strategy (The "Brain" of the network)
# We use FedProx to handle Non-IID data distributions.
strategy = fl.server.strategy.FedProx(
    fraction_fit=1.0,           # Use 100% of available clients for training
    fraction_evaluate=1.0,      # Use 100% of available clients for testing
    min_fit_clients=2,          # Wait for at least 2 clients to connect
    min_evaluate_clients=2,     
    min_available_clients=2,
    proximal_mu=0.1,            # The penalty term to prevent model drift
)

# 2. Main Execution
if __name__ == "__main__":
    print("==============================================")
    print("FEDERATED LEARNING SERVER: FEDPROX ENABLED")
    print("==============================================")
    
    # Start the Flower server on port 8080
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3), # 3 Rounds for the demo
        strategy=strategy,
    )