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


def get_good_trials(matlab_file):
    # TODO: this is specific to A026-20200323-01.mat and is used to filter out trials with different cue patterns
    
    m = loadmat(matlab_file, simplify_cells=True) # raw data from matlab
    return m
    idx = m['Track']['lapID'].astype(int)

    mov = m["Laps"]["movieOnLfpInd"] # movie on times
    cnt = np.array([ len(i) for i in mov ]) # get number of movie on times per trial
    good = np.where(cnt == 3)[0] # indices for standard trials

    n = np.unique(idx[good])

    return n


def get_exp_data(matlab_file):
    '''extracts experimental data from single ephys session
    
    requires path to matlab file

    returns single dataframe'''

    m = loadmat(matlab_file, simplify_cells=True) # raw data from matlab
    df = pd.DataFrame() # empty dataframe

    # trial indices
    df.loc[:, 'trial'] = m['Track']['lapID'].astype(int)
    g = np.gradient(df.loc[:, 'trial']) # change index after last trial to -1 (this may not be consisent across sessions)
    x = np.where( g < 0 )[0][0] # index where lapID is about to decrease
    df.loc[x+1:, 'trial'] = -1
    ntot = len(df.loc[:, 'trial']) # total number of time steps

    # distance
    df.loc[:, 'dst'] = m["Track"]["xMM"].astype(float)

    # licks
    lck = convert_train(m["Laps"]["lickLfpInd"] , ntot) 
    df.loc[:, 'lck'] = lck.astype(int)

    # rewards
    pmp = m["Laps"]["pumpLfpInd"] # pump on/off
    pmp0 = np.array([i[0] for i in pmp]) # only consider pump on
    rwd = convert_train(pmp0, ntot)
    df.loc[:, 'rwd'] = rwd.astype(int)

    # cue and blackout
    mov = m["Laps"]["movieOnLfpInd"] # movie times: cue on, cue off, blk on
    mov0 = np.array([i[0] for i in mov]) # first element: cue on
    cue_on = convert_train(mov0, ntot) 
    df.loc[:, 'cue_on'] = cue_on.astype(int)

    mov1 = np.array([i[1] for i in mov]) # second element: cue off
    cue_off = convert_train(mov1, ntot) 
    df.loc[:, 'cue_off'] = cue_off.astype(int)

    mov2 = np.array([ i[2] for i in mov ]) # third element is blackout on 
    blk_on = convert_train(mov2, ntot)
    df.loc[:, 'blk_on'] =  blk_on.astype(int)

    nxt = m["Laps"]["startLfpInd"] # start trial?
    nxt = np.array([ i - 1 for i in nxt[1:]]) # next trial is blk off 
    blk_off = convert_train(nxt, ntot) 
    df.loc[:, 'blk_off'] = blk_off.astype(int)

    # extra
    # TODO: this is specific to A026-20200323-01.mat and is used to filter out trials with different cue patterns
    xtr = np.array([ i[3] for i in mov if len(i) == 4])
    xtr = convert_train(xtr, ntot) 
    df.loc[:, 'xtr'] = xtr.astype(int)

    # spikes 
    spk_raw = m["Spike"]["res"] # all spike times
    clu = m["Spike"]["totclu"] # corresponding clusters

    for i in np.unique(clu):
        s_raw = spk_raw[np.where(clu == i )[0]] # spikes from single single cluster
        spk = convert_train(s_raw, ntot)
        
        n = 'unt_{}'.format(str(i)) # name for cluster
        df.loc[:, n] = spk.astype(int)

    return df    

