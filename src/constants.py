POSITION_LIMIT = 16
SCALE_LIMIT = 6
ROTATION_LIMIT = 60
ROTATION_DELTA = 180 / ROTATION_LIMIT

BETA_LOW = 0.0
BETA_HIGH = 5.0
BETA_STEP = 0.20

NOISE = 0.20

# Architecture codes.
FC = 0
CONV = 1

def run_index():
    with open('run_counter.txt', 'r') as f:
        return map(int, f)[0]

def increase_index():
    idx = run_index() + 1
    with open('run_counter.txt', 'r+') as f:
        f.write(str(idx))
        f.truncate()

