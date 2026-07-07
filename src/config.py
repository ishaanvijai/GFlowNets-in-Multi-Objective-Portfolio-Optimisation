# Hyperparameters and constants

N = 100          # Number of assets in the universe
K = 3
STEP = 0.01      # Weight granularity (1% increments)
MAXWEIGHT = 0.50
NUMSAMPLES = 1000  # Default samples per evaluation run

# Training
BATCH = 8
N_ITER = 300
LR_POLICY = 0.000495206
LR_LOGZ = 0.029297

# Auxiliary GFlowNet (LGGFN)
LAMBDA_AUX = 0.6571
AUX_BATCH = 12

# Date range
START_DATE = "2015-09-01"
END_DATE = "2025-09-01"
TRAIN_END = "2023-08-31"
TEST_START = "2023-09-01"

# Walk-forward validation windows: (train_end, dev_start, dev_end)
# Expanding window — training always starts from START_DATE.
# The dev periods partition 2020-2023, leaving 2023-09-01 onward as the
# untouched test set.
WF_WINDOWS = [
    ("2019-12-31", "2020-01-01", "2020-12-31"),
    ("2020-12-31", "2021-01-01", "2021-12-31"),
    ("2021-12-31", "2022-01-01", "2022-12-31"),
    ("2022-12-31", "2023-01-01", "2023-08-31"),
]
