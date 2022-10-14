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
    idx = idx - 1 # start counting at 0

    arr = np.zeros(ntot) # empty array with same size as xMM
    arr[idx] = 1 # set lick events as 1

    return arr


def get_exp_data(matlab_file):
    '''Process matlab file stored on disk containing ephys experiment data

    Returns dictionary with the experimental features:

    read in ephys data stored as matlab struct

    returns dict of experimental features'''

    m = loadmat(matlab_file, simplify_cells=True) # raw data from matlab

    # get start indices for trials
    lap_idx = m['Track']['lapID']
    trial_idx = np.unique(lap_idx) # array of unique trial ids
    trial_idx = np.nonzero(trial_idx)[0] # trials start at 1

    trial_start_idx = [] # list with trial start indices
    for i in trial_idx:
        s = np.where(lap_idx == i)[0] # index for first occurence of i
        trial_start_idx.append(s[0])
    else:
        trial_start_idx.append(s[-1] + 1) # one after end of last trial
    trial_start_idx = np.array(trial_start_idx)

    # get total number of time steps
    dst_raw = m["Track"]["xMM"]
    ntot = len(dst_raw)

    # split array of len=ntot into trials and drop data before first and after last trial
    split_trials = lambda arr: np.split(arr, trial_start_idx)[1:-1] 

    # distance
    dst = split_trials(dst_raw)

    # licks
    lck_raw = m["Laps"]["lickLfpInd"] 
    lck = convert_train(lck_raw, ntot) 
    lck = split_trials(lck)    

    # rewards
    rwd_raw = m["Laps"]["pumpLfpInd"]

    rwd_on_raw = np.array([i[0] for i in rwd_raw])
    rwd_on = convert_train(rwd_on_raw, ntot) 
    rwd_on = split_trials(rwd_on)

    rwd_off_raw = np.array([i[1] for i in rwd_raw])
    rwd_off = convert_train(rwd_off_raw, ntot) 
    rwd_off = split_trials(rwd_off) 

    # standard and non-standard trials 
    # TODO: this is specific to A026-20200323-01.mat and is used to filter out trials with different cue patterns
    mov_raw = m["Laps"]["movieOnLfpInd"] # movie on times
    mov_counts = np.array([ len(i) for i in mov_raw ]) # get number of movie on times per trial
    std_trial_idx = np.where(mov_counts == 3)[0] # indices for standard trials

    # cues
    cue_on_raw = np.array([i[0] for i in mov_raw]) # first two elements are cue on/off
    cue_on = convert_train(cue_on_raw, ntot) 
    cue_on = split_trials(cue_on)

    cue_off_raw = np.array([i[1] for i in mov_raw]) # first two elements are cue on/off
    cue_off = convert_train(cue_off_raw, ntot) # original code did idx-1 for cue off
    cue_off = split_trials(cue_off)

    # blackout
    blk_on_raw = np.array([ i[2] for i in mov_raw ]) # third element is blk on 
    blk_on = convert_train(blk_on_raw, ntot) 
    blk_on = split_trials(blk_on)

    blk_off_raw = trial_start_idx[1:] # next trial is blk off
    blk_off = convert_train(blk_off_raw, ntot) 
    blk_off = split_trials(blk_off)

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
        spk = convert_train(s_raw, ntot)
        spk = split_trials(spk)

        n = 'cluster_{}'.format(str(i)) # name for cluster
        raw_spikes[n] = spk

    return raw_behavior, raw_spikes, std_trial_idx
    
