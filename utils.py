import numpy as np
import pandas as pd
from scipy.io import loadmat


def steps2ms(x):
    'convert raw time steps to ms'
    y = x / 1250 * 1000 # "raw" data is at 1250 Hz
    return y

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


def get_std_trials(matlab_file):
    # TODO: this is specific to A026-20200323-01.mat and is used to filter out trials with different cue patterns
    
    m = loadmat(matlab_file, simplify_cells=True) # raw data from matlab

    mov_raw = m["Laps"]["movieOnLfpInd"] # movie on times
    mov_counts = np.array([ len(i) for i in mov_raw ]) # get number of movie on times per trial
    std_trial_idx = np.where(mov_counts == 3)[0] # indices for standard trials


def get_exp_data(matlab_file):
    '''Process matlab file stored on disk containing ephys experiment data

    returns single dataframe'''

    m = loadmat(matlab_file, simplify_cells=True) # raw data from matlab
    df = pd.DataFrame(dtype=float) # empty dataframe

    # trial indices
    df.loc[:, 'trial'] = m['Track']['lapID'] # TODO change index of after last trial to something else then 0
    ntot = len(df.loc[:, 'trial']) # total number of time steps

    # distance
    df.loc[:, 'dst'] = m["Track"]["xMM"]

    # licks
    df.loc[:, 'lck'] = convert_train(m["Laps"]["lickLfpInd"] , ntot) 

    # rewards
    pmp = m["Laps"]["pumpLfpInd"] # pump on/off
    rwd = np.array([i[0] for i in pmp]) # only consider pump on
    df.loc[:, 'rwd'] = convert_train(rwd, ntot) 

    # cue and blackout
    # TODO: this is specific to A026-20200323-01.mat and is used to filter out trials with different cue patterns
    mov = m["Laps"]["movieOnLfpInd"] # movie times: cue on, cue off, blk on
    cue_on = np.array([i[0] for i in mov]) # first element: cue on
    df.loc[:, 'cue_on'] = convert_train(cue_on, ntot) 

    cue_off = np.array([i[1] for i in mov]) # second element: cue off
    df.loc[:, 'cue_off'] = convert_train(cue_off, ntot) 

    blk_on = np.array([ i[2] for i in mov ]) # third element is blackout on 
    df.loc[:, 'blk_on'] = convert_train(blk_on, ntot) 

    nxt = m["Laps"]["startLfpInd"] # start trial?
    blk_off = np.array([ i - 1 for i in nxt[1:]]) # next trial is blk off 
    df.loc[:, 'blk_off'] = convert_train(blk_off, ntot) 

    # spikes 
    spk_raw = m["Spike"]["res"] # all spike times
    clu = m["Spike"]["totclu"] # corresponding clusters

    for i in np.unique(clu):
        s_raw = spk_raw[np.where(clu == i )[0]] # spikes from single single cluster
        spk = convert_train(s_raw, ntot)
        
        n = 'unt_{}'.format(str(i)) # name for cluster
        df.loc[:, n] = spk 

    return df    
