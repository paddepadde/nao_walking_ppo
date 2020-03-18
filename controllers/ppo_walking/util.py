from __future__ import division
import numpy as np

def detect_fall(pos):
    if pos[1] < 0.2:
        return True
    else:
        return False    

# rescales value from interval (-1, 1) into (a, b)
def rescale_in_range(val, a=-1.0, b=1.0):
    new_value = (b-a) * ( (val + 1) / (1 + 1)) + a
    return new_value

def compute_std(step):
    return np.clip(-3e-7 * step - 0.7, -1.6, -0.7)
