import flwr as fl

# This is a dummy client to test the server connection
class TestClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        print("Connected! Server requested training.")
        return [], 1, {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

if __name__ == "__main__":
    # Connect to the server you have running on port 8080
    fl.client.start_client(
        server_address="127.0.0.1:8080", 
        client=TestClient().to_client()
    )