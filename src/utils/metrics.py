import numpy as np


# ---------------------------------------------------
# JAIN FAIRNESS INDEX
# ---------------------------------------------------

def jain_fairness_index(values):
    """
    Jain's Fairness Index:
    Measures how equally contributions are distributed.

    Range:
    0 → worst fairness
    1 → perfect fairness
    """
    values = np.array(values, dtype=np.float32)

    if len(values) == 0:
        return 0.0

    numerator = np.sum(values) ** 2
    denominator = len(values) * np.sum(values ** 2)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


# ---------------------------------------------------
# ENERGY CONSUMPTION
# ---------------------------------------------------

def energy_consumption(before, after):
    """
    Calculates battery usage
    """
    return max(before - after, 0)


# ---------------------------------------------------
# NORMALIZATION HELPERS (FOR RL STATE)
# ---------------------------------------------------

def normalize_battery(battery):
    return battery / 100.0


def normalize_latency(latency):
    return 1.0 / (latency + 1.0)


def normalize_reliability(reliability):
    return reliability