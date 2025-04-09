import math
from typing import List

N: int = 3
FLAT_DIM: int = N * N
ACTION_DIM: int = FLAT_DIM
MAX_TRAJECTORY_LEN: int = N * N + 10
MAX_THEORETICAL_STEPS: int = N * N
REWARD_EXP_MULTIPLIER: float = 0.8
MIN_REWARD_LOG: float = -20.0
MIN_REWARD: float = math.exp(MIN_REWARD_LOG)

HIDDEN_DIM: int = 128

LR: float = 1e-4
WEIGHT_DECAY: float = 1e-4
WARMUP_STEPS: int = 1000
BATCH_SIZE: int = 512
TRAINING_STEPS: int = 100000

LOG_EVERY: int = 100
EVAL_EVERY: int = 1000
EVAL_SAMPLES_PER_K: int = 64
EVAL_MAX_STEPS: int = MAX_THEORETICAL_STEPS * 3
EVAL_K_VALUES: List[int] = list(range(1, N * N + 1))

MLFLOW_URI: str = "file:./mlruns"
MLFLOW_EXP: str = "GFN_LightsOut_N3_Compare_NNX_MLP"
