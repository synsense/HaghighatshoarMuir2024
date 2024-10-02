import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

data_directory = Path('demo-benchmark-simulation-freq1600-2000')
data_directory = Path('demo-benchmark-simulation-freq2000-2300')
data_directory = Path('demo-benchmark-simulation-freq2300-2600')

data_files = list(data_directory.glob('**/*.txt'))

data = np.loadtxt(data_files[2])

from copy import deepcopy

def make_window(source, win_length):
    ret_data = np.empty((np.size(source) - win_length, win_length))
    for wind_ind in range(win_length):
        ret_data[:, wind_ind] = source[wind_ind:-(win_length-wind_ind)]

    return ret_data

def remove_jumps(source, jump_reject):
    filter = np.concatenate((np.diff(source) > jump_reject, [False], ))
    data_ret = deepcopy(source)
    data_ret[filter] = np.nan
    return data_ret
    
def remove_jumps_fore(source, jump_reject):
    filter = np.concatenate(([False], np.diff(source) > jump_reject))
    data_ret = deepcopy(source)
    data_ret[filter] = np.nan
    return data_ret

def remove_from_med(source, reject_dist):
    filter = np.abs(source - np.median(source)) > reject_dist
    data_ret = deepcopy(source)
    data_ret[filter] = np.nan
    return data_ret

def window_median(source, window_length, reject_jump):
    source_window = make_window(source, window_length)
    length = np.shape(source_window)[0]
    data_ret = np.empty(length)

    for wind_ind in range(length):
        this_wind = source_window[wind_ind, :]
        diff = this_wind - np.median(this_wind)
        this_wind[np.abs(diff > reject_jump)] = np.nan
        data_ret[wind_ind] = np.nanmedian(this_wind)

    return data_ret

def window_mean(source, window_length, reject_jump):
    source_window = make_window(source, window_length)
    length = np.shape(source_window)[0]
    data_ret = np.empty(length)

    for wind_ind in range(length):
        this_wind = source_window[wind_ind, :]
        diff = this_wind - np.median(this_wind)
        this_wind[np.abs(diff > reject_jump)] = np.nan
        data_ret[wind_ind] = np.nanmean(this_wind)

    return data_ret

def post_process_data(source_dir, window_length, jump_reject = 20):
    data_files = list(Path(source_dir).glob('**/*.txt'))
    source_data = np.loadtxt(data_files[2])

    return window_median(source_data, window_length, jump_reject)

def mae(source, target):
    return np.mean(np.abs(source - target))

print(
    mae(post_process_data('demo-benchmark-simulation-freq1600-2000', 25), 128.571429),
    mae(post_process_data('demo-benchmark-simulation-freq2000-2300', 25), 129.375),
    mae(post_process_data('demo-benchmark-simulation-freq2300-2600', 25), 132.589286)
)


