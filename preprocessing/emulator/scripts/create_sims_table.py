# python
import os
import itertools

# external
import numpy as np
import pandas as pd

# =============================================================================
# 1. conditions
# =============================================================================

## complete range
water_level_list          = [1, 2, 3]
wind_speed_list           = [10, 20, 30]
wind_dir_list             = [15, 30, 45]


# =============================================================================
# 2. make combinations
# =============================================================================
comb         = [wind_dir_list, wind_speed_list, water_level_list]
combinations = list(itertools.product(*comb))


rows = []
for (wind_dir, wind_speed, water_level) in combinations:
    row = {"wind_dir": wind_dir, "wind_speed": wind_speed, "water_level": water_level}
    rows.append(row)

df = pd.DataFrame(rows)
df.to_hdf('runs.h5', 'runs')

print(f'Created {df.shape[0]} runs')
