import flwr as fl
import numpy as np

# RL import
try:
    from src.rl.agent import RLAgent
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Metrics
from src.utils.metrics import (
    jain_fairness_index,
    normalize_battery,
    normalize_latency,
    normalize_reliability,
)


class RLWeightedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_clients=5):
        super().__init__()

        self.round = 0
        self.num_clients = num_clients
        self.global_accuracy = 0.7

        if RL_AVAILABLE:
            print("[SERVER] RL Agent detected")
            self.agent = RLAgent(num_clients=num_clients)
            self.rl_ready = True
        else:
            print("[SERVER] RL NOT available → using heuristic")
            self.rl_ready = False

    # ---------------------------------------------------
    # AGGREGATION
    # ---------------------------------------------------

    def aggregate_fit(self, rnd, results, failures):
        self.round += 1

        print("\n" + "=" * 60)
        print(f"[SERVER] ROUND {rnd}")
        print("=" * 60)

        if not results:
            print("[SERVER] No results received")
            return None, {}

        # ---------------------------------------------------
        # FILTER INVALID CLIENTS (🔥 CRITICAL FIX)
        # ---------------------------------------------------

        valid_results = []

        for client, fit_res in results:
            if (
                fit_res.parameters is None
                or len(fit_res.parameters.tensors) == 0
            ):
                print("[SERVER] Skipping empty client update")
                continue

            valid_results.append((client, fit_res))

        if len(valid_results) == 0:
            print("[SERVER] No valid client updates")
            return None, {}

        results = valid_results
        num_clients = len(results)

        # ---------------------------------------------------
        # COLLECT DATA
        # ---------------------------------------------------

        batteries = []
        latencies = []
        reliabilities = []
        params_list = []

        for _, fit_res in results:
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)

            battery = fit_res.metrics.get("battery", 50.0)
            latency = fit_res.metrics.get("latency", 100.0)
            reliability = fit_res.metrics.get("reliability", 0.9)

            batteries.append(battery)
            latencies.append(latency)
            reliabilities.append(reliability)
            params_list.append(params)

        # ---------------------------------------------------
        # BUILD STATE
        # ---------------------------------------------------

        avg_battery = normalize_battery(np.mean(batteries))
        avg_latency = normalize_latency(np.mean(latencies))
        avg_reliability = normalize_reliability(np.mean(reliabilities))
        accuracy = self.global_accuracy

        state = np.array(
            [avg_battery, avg_latency, avg_reliability, accuracy],
            dtype=np.float32,
        )

        print(f"[STATE] {state}")

        # ---------------------------------------------------
        # GET WEIGHTS (RL or fallback)
        # ---------------------------------------------------

        if self.rl_ready:
            try:
                action = self.agent.predict_weights(state)

                weights = np.array(action)

                print("[RL] Using RL weights")

            except Exception as e:
                print(f"[RL ERROR] {e}")
                weights = self._heuristic_weights(
                    batteries, latencies, reliabilities
                )
        else:
            weights = self._heuristic_weights(
                batteries, latencies, reliabilities
            )

        # ---------------------------------------------------
        # 🔥 FIX: MATCH WEIGHTS TO VALID CLIENTS
        # ---------------------------------------------------

        weights = weights[:num_clients]

        # Avoid zero division
        weights = np.clip(weights, 1e-6, None)
        weights = weights / np.sum(weights)

        print(f"[WEIGHTS] {weights}")

        # ---------------------------------------------------
        # FAIRNESS
        # ---------------------------------------------------

        fairness = jain_fairness_index(weights)
        print(f"[FAIRNESS] {fairness:.4f}")

        # ---------------------------------------------------
        # REWARD (for logging)
        # ---------------------------------------------------

        reward = accuracy - (0.5 * (1 - avg_battery)) + fairness
        print(f"[REWARD] {reward:.4f}")

        # ---------------------------------------------------
        # SAFE AGGREGATION (🔥 FIXED LOOP)
        # ---------------------------------------------------

        aggregated_params = []
        num_layers = len(params_list[0])

        for layer_idx in range(num_layers):
            layer_sum = np.zeros_like(params_list[0][layer_idx])

            for i in range(len(params_list)):  # SAFE LOOP
                layer_sum += params_list[i][layer_idx] * weights[i]

            aggregated_params.append(layer_sum)

        print("[SERVER] Aggregation complete\n")

        return fl.common.ndarrays_to_parameters(aggregated_params), {}

    # ---------------------------------------------------
    # HEURISTIC (fallback)
    # ---------------------------------------------------

    def _heuristic_weights(self, batteries, latencies, reliabilities):
        print("[HEURISTIC] Using fallback")

        weights = []

        for b, l, r in zip(batteries, latencies, reliabilities):
            w = (b / 100.0) * (1.0 / (l + 1.0)) * r
            weights.append(w)

        return np.array(weights)