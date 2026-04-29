import flwr as fl
import os
import sys
from typing import List, Tuple

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------
# IMPORT STRATEGY
# ---------------------------------------------------

from src.strategy.rl_weighted import RLWeightedStrategy

# ---------------------------------------------------
# CONFIG (can later move to config.yaml)
# ---------------------------------------------------

NUM_ROUNDS = 3


# ---------------------------------------------------
# CUSTOM STRATEGY WRAPPER
# ---------------------------------------------------

class ServerWithEval(RLWeightedStrategy):
    """
    Extends RLWeightedStrategy to:
    - capture evaluation metrics
    - update global accuracy for RL state
    """

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures,
    ):
        if not results:
            return None, {}

        accuracies = []
        losses = []

        for _, eval_res in results:
            acc = eval_res.metrics.get("accuracy", 0.0)
            accuracies.append(acc)
            losses.append(eval_res.loss)

        avg_acc = sum(accuracies) / len(accuracies)
        avg_loss = sum(losses) / len(losses)

        # 🔥 Update RL state (VERY IMPORTANT)
        self.global_accuracy = avg_acc

        print("\n" + "-" * 60)
        print(f"[EVAL] Round {rnd}")
        print(f"[EVAL] Avg Accuracy: {avg_acc:.4f}")
        print(f"[EVAL] Avg Loss: {avg_loss:.4f}")
        print("-" * 60 + "\n")

        return super().aggregate_evaluate(rnd, results, failures)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("🚀 SELF-EVOLVING FL SERVER STARTED")
    print("=" * 60)

    print(f"[PATH] Project Root: {PROJECT_ROOT}")
    print("[INFO] Waiting for clients on 127.0.0.1:8080...\n")

    # Initialize strategy
    strategy = ServerWithEval(num_clients=5)

    # 🔥 OPTIONAL: Pre-train RL agent (good for demo stability)
    if hasattr(strategy, "agent"):
        print("[RL] Pre-training agent...")
        try:
            strategy.agent.train(timesteps=1000)
        except Exception as e:
            print(f"[RL WARNING] Training skipped: {e}")

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------

if __name__ == "__main__":
    main()