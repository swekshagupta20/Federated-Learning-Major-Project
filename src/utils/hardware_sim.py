import json
import os
import numpy as np

class HardwareManager:
    def __init__(self, metadata_path='data/partition_metadata.json'):
        # Find the JSON file regardless of where we run the script
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.path = os.path.join(base_dir, metadata_path)
        
        with open(self.path, 'r') as f:
            self.data = json.load(f)

    def get_sensor_data(self, client_id):
        """Returns the current state of a specific client's hardware."""
        state = self.data[str(client_id)]['system_state']
        # Return as a list: [Battery, Latency, Reliability]
        return [state['battery'], state['latency'], state['reliability']]

    def simulate_drain(self, client_id, energy_cost=5.0):
        """Reduces battery after a training round."""
        self.data[str(client_id)]['system_state']['battery'] -= energy_cost
        # Prevent negative battery
        if self.data[str(client_id)]['system_state']['battery'] < 0:
            self.data[str(client_id)]['system_state']['battery'] = 0
            
        # Save the updated state back to the JSON
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=4)