import math
from typing import List

N: int = 3
FLAT_DIM: int = N * N
ACTION_DIM: int = FLAT_DIM
MAX_TRAJECTORY_LEN: int = N * N + 10 # e.g., 19 for N=3
MAX_THEORETICAL_STEPS: int = N * N # Max unique non-repeating steps

# --- Reward Configuration (Inverse Step Reward) ---
REWARD_INV_STEPS_C: float = 0.05 # Slower decay
MIN_REWARD_LOG: float = -15.0
MIN_REWARD: float = math.exp(MIN_REWARD_LOG) # Automatically calculated

# --- Model & Training Config ---
HIDDEN_DIM: int = 256
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-2
WARMUP_STEPS: int = 1000
BATCH_SIZE: int = 512
TRAINING_STEPS: int = 20000

LOG_EVERY: int = 100
EVAL_EVERY: int = 1000
EVAL_SAMPLES_PER_K: int = 64 # For K-perturb eval
EVAL_MAX_STEPS: int = MAX_TRAJECTORY_LEN # Max steps for greedy/stochastic eval solvers
EVAL_K_VALUES: List[int] = list(range(1, N * N + 1)) # K=1 to 9
EVAL_STOCHASTIC_RUNS_PERIODIC: int = 10 # Runs during training eval
EVAL_STOCHASTIC_RUNS_FINAL: int = 20   # Runs during final eval

MLFLOW_URI: str = "file:./mlruns"
MLFLOW_EXP: str = "GFN_LightsOut_N3"
