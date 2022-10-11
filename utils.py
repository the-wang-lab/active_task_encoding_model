import numpy as np
import pandas as pd
from scipy.io import loadmat

def downsample_arr(arr, interval, mode='average'):
    '''Downsample 1D array by the int given by interval

    Choose a mode depending on the data:
    continuous variables: average
    delta-function-like events: sum
    square-wave-type intervals: ones
    
    Returns downsampled 1D array'''

    # append np.nan so array shape correct for reshape
    miss_val = (interval - (len(arr) % interval)) % interval
    arr = np.append(arr, np.repeat(np.nan,  miss_val))

    # reshape to 2D
    arr = np.reshape(arr, (-1, interval))

    # collapse to 1D
    if mode == 'sum':
        arr = arr.sum(axis=1)
    elif mode == 'average':
        arr = arr.mean(axis=1)
    elif mode == 'ones':
        arr = arr.mean(axis=1)
        arr[ arr != 0 ] = 1

    return arr


def convert_train(raw, ntot):
    '''Create 1D array from event times
    
    raw is (potentially nested) array with integers indicating event times (starts counting at 1)
    ntot is the total lengths of new 1D array
    
    Returns 1D array with 1 at the index of the event, otherwise 0'''

    try:
        idx = np.concatenate(raw) # flatten array
    except ValueError: # array appears to be already flattened
        idx = raw
    idx -= 1 # start counting at 0

    arr = np.zeros(ntot) # empty array with same size as xMM
    arr[idx] = 1 # set lick events as 1

    return arr


def get_exp_data(matlab_file, downsample=1):
    '''Process matlab file stored on disk containing ephys experiment data

    The time steps are downsampled to the interval given by the integer downsample

    Returns dictionary with the experimental features:

    read in ephys data stored as matlab struct
    downsample by bin_size

    returns dict of experimental features'''

    m = loadmat(matlab_file, simplify_cells=True) # raw data from matlab

    ntot = len(m["Track"]["xMM"]) # total number of time steps

    # distance
    dst = m["Track"]["xMM"]
    dst = downsample_arr(dst, interval=downsample, mode='average')

    # licks
    lck_raw = m["Laps"]["lickLfpInd"] 
    lck = convert_train(lck_raw, ntot) 
    lck = downsample_arr(lck, interval=downsample, mode='sum') # downsample and sum
    
    # rewards
    rwd_raw = m["Laps"]["pumpLfpInd"]

    rwd_on_raw = np.array([i[0] for i in rwd_raw])
    rwd_on = convert_train(rwd_on_raw, ntot) 
    rwd_on = downsample_arr(rwd_on, interval=downsample, mode='sum') 

    rwd_off_raw = np.array([i[1] for i in rwd_raw])
    rwd_off = convert_train(rwd_off_raw, ntot) 
    rwd_off = downsample_arr(rwd_off, interval=downsample, mode='sum') 

    # cue and blackout
    mov_raw = m["Laps"]["movieOnLfpInd"] # movie on times?

    cue_on_raw = np.array([i[0] for i in mov_raw]) # first two elements are cue on/off
    cue_on = convert_train(cue_on_raw, ntot) 
    cue_on = downsample_arr(cue_on, interval=downsample, mode='sum')

    cue_off_raw = np.array([i[1] for i in mov_raw]) # first two elements are cue on/off
    cue_off = convert_train(cue_off_raw, ntot) # original code did idx-1 for cue off
    cue_off = downsample_arr(cue_off, interval=downsample, mode='sum')

    blk_on_raw = np.array([ i[2] for i in mov_raw ]) # third element is blk on (why do some have 4 elements?)
    blk_on = convert_train(blk_on_raw, ntot) 
    blk_on = downsample_arr(blk_on, interval=downsample, mode='sum')

    str_raw = m["Laps"]["startLfpInd"] # start trial?
    blk_off_raw = np.array([ j for j in str_raw[1:]]) # next trial is blk off 
    blk_off = convert_train(blk_off_raw, ntot)  # original code did idx-1 for blk off
    blk_off = downsample_arr(blk_off, interval=downsample, mode='sum')

    # compile to dict
    raw_behavior = {
        'distance': dst,
        'licks':    lck,
        'reward_on':   rwd_on,
        'reward_off':   rwd_off,
        'cue_on':   cue_on,
        'cue_off':  cue_off,
        'blackout_on': blk_on,
        'blackout_off': blk_off,
    }

    # spikes 
    spk_raw = m["Spike"]["res"] # all spike times
    clu = m["Spike"]["totclu"] # corresponding clusters

    raw_spikes = dict() # dict with one spikes array per cluster
    for i in np.unique(clu):
        s_raw = spk_raw[np.where(clu == i )[0]] # spkikes from single single cluster
        s = convert_train(s_raw, ntot) 
        spk = downsample_arr(s, interval=downsample, mode='sum') # downsample and sum

        n = 'cluster_{}'.format(str(i)) # name for cluster
        raw_spikes[n] = spk

    return raw_behavior, raw_spikes
