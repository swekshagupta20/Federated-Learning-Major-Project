import json
import os
import numpy as np


class HardwareManager:
    """
    Simulates edge device conditions:
    - Battery
    - Latency
    - Reliability (drop probability)
    """

    def __init__(self, metadata_path="data/partition_metadata.json"):
        # Resolve project root safely
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.path = os.path.join(base_dir, metadata_path)

        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Metadata file not found at {self.path}. Run partition_data.py first."
            )

        self._load()

    # ---------------------------------------------------
    # LOAD / SAVE
    # ---------------------------------------------------

    def _load(self):
        with open(self.path, "r") as f:
            self.data = json.load(f)

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=4)

    # ---------------------------------------------------
    # GET STATE
    # ---------------------------------------------------

    def get_sensor_data(self, client_id):
        state = self.data[str(client_id)]["system_state"]

        return (
            float(state["battery"]),
            float(state["latency"]),
            float(state["reliability"]),
        )

    # ---------------------------------------------------
    # BATTERY DRAIN
    # ---------------------------------------------------

    def simulate_drain(self, client_id, energy_cost=5.0):
        state = self.data[str(client_id)]["system_state"]

        state["battery"] -= energy_cost
        state["battery"] = max(state["battery"], 0.0)

        self._save()

    # ---------------------------------------------------
    # NETWORK FLUCTUATION
    # ---------------------------------------------------

    def simulate_network(self, client_id):
        state = self.data[str(client_id)]["system_state"]

        # Latency noise
        latency_noise = np.random.normal(0, 5)
        state["latency"] += latency_noise

        # Clamp latency
        state["latency"] = float(np.clip(state["latency"], 5, 500))

        # Reliability fluctuation
        reliability_noise = np.random.normal(0, 0.02)
        state["reliability"] += reliability_noise

        # Clamp reliability
        state["reliability"] = float(np.clip(state["reliability"], 0.5, 1.0))

        self._save()

    # ---------------------------------------------------
    # DROPOUT SIMULATION
    # ---------------------------------------------------

    def is_available(self, client_id):
        """
        Returns True if client participates,
        False if client drops out.
        """
        state = self.data[str(client_id)]["system_state"]
        reliability = state["reliability"]

        return np.random.rand() < reliability