import flwr as fl
import os
import sys

# 1. Dynamic Path Setup
# This finds the 'major 2 work' root folder automatically
current_file = os.path.abspath(__file__) # E:\major 2 work\src\server\server.py
server_dir = os.path.dirname(current_file) # E:\major 2 work\src\server
src_dir = os.path.dirname(server_dir) # E:\major 2 work\src
project_root = os.path.dirname(src_dir) # E:\major 2 work

# Add the root to sys.path so 'src' can be seen as a module
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. Strategy Definition
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=1,
    min_available_clients=1,
)

if __name__ == "__main__":
    print("\n" + "="*45)
    print("      FL SERVER: PHASE 2 (MODULAR) ONLINE")
    print("="*45)
    print(f"[PATH] Root detected at: {project_root}")
    print("[INFO] Waiting for client connections on port 8080...")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )