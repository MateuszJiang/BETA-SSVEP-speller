import constants
import os
import numpy as np
import mne
from os.path import dirname, join as pjoin
import scipy.io as sio
import csv

def wholeDatabaseReader():
    TrialMatrix = np.zeros((constants.DATABASE_TRIALS, constants.DATABASE_SUBJECTS, constants.BLOCKS, 
                            constants.DATABASE_NUMBER_OF_CHANNELS, constants.NUMBER_OF_SAMPLES))

    # The aim is to make 70x4 matrix of arrays with measurements 64x1000.
    for t in range(0, constants.DATABASE_TRIALS):
        data_dir = pjoin('dir_trial_' + str(t+1))
        for s in range(0 , constants.DATABASE_SUBJECTS):
            for b in  range(0 , constants.BLOCKS):
                fineName = 'subject_' + str(s+1) + '_block_' + str(b+1) + '.mat'
                mat_fname = pjoin(data_dir, fineName)
                mat_contents = sio.loadmat(mat_fname)
                TrialMatrix[t,s,b] = mat_contents['results']
                
    return TrialMatrix

def myMeasurementsReader(key_directory):
    TrialMatrix = np.zeros((constants.MY_TRIALS, constants.MY_SUBJECTS, constants.BLOCKS, constants.MY_NUMBER_OF_CHANNELS, 
                            constants.NUMBER_OF_SAMPLES))
    
    for t in range(0, constants.MY_TRIALS):
        trial_name = 'dir_trial_' + str(t+1)
        data_dir = os.path.join(key_directory, trial_name)
        for s in range(0 , constants.MY_SUBJECTS):
            for b in  range(0 , constants.BLOCKS):
                fineName = 'subject_' + str(s+1) + '_block_' + str(b+1) + '.mat'
                mat_fname = pjoin(data_dir, fineName)
                mat_contents = sio.loadmat(mat_fname)
                TrialMatrix[t,s,b] = mat_contents['measuredSignal']
                
    return TrialMatrix
                
                
def myMeasurements_fif_reader(key_directory):
    TrialMatrix = np.zeros((constants.MY_TRIALS, constants.MY_SUBJECTS, constants.BLOCKS, constants.MY_NUMBER_OF_CHANNELS, 
                            constants.NUMBER_OF_SAMPLES))
    
    for t in range(0 , constants.MY_TRIALS):
        trial_name = 'dir_trial_' + str(t+1)
        data_dir = os.path.join(key_directory, trial_name)
        for s in range(0 , constants.MY_SUBJECTS):
            for b in  range(0 , constants.BLOCKS):
                fineName = 'subject_' + str(s+1) + '_block_' + str(b+1) + '.csv'
                mat_fname = pjoin(data_dir, fineName)
                data = np.loadtxt(open(mat_fname, "rb"), delimiter=",")
                TrialMatrix[t,s,b] = data
                
    return TrialMatrix
                

def databaseMeasurements_fif_reader(key_directory):
    TrialMatrix = np.zeros((constants.DATABASE_TRIALS, constants.DATABASE_SUBJECTS, constants.BLOCKS, 
                            constants.DATABASE_NUMBER_OF_CHANNELS, constants.NUMBER_OF_SAMPLES))

    for t in range(0 , constants.DATABASE_TRIALS):
        trial_name = 'dir_trial_' + str(t+1)
        data_dir = pjoin(key_directory, trial_name)
        for s in range(0 , constants.DATABASE_SUBJECTS):
            for b in  range(0 , constants.BLOCKS):
                fineName = 'subject_' + str(s+1) + '_block_' + str(b+1) + '.csv'
                mat_fname = pjoin(data_dir, fineName)
                data = np.loadtxt(open(mat_fname, "rb"), delimiter=",")
                TrialMatrix[t,s,b] = data
                
    return TrialMatrix
