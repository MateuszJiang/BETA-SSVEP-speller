import os
import numpy as np
import mne
from os.path import dirname, join as pjoin
import scipy.io as sio
import shutil

import constants
import eeg_data_readers


myTrials = eeg_data_readers.myMeasurementsReader('my_performed_measurements')
info = mne.create_info(constants.MY_CH_NAMES, constants.SAMPLING_FREQUENCY, ch_types='eeg')
    
singleTrial = myTrials[1,0,0]

results = np.zeros(constants.DATABASE_TRIALS)
processed_dir = 'my_processed_measurements'
if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
os.mkdir(processed_dir)

for t in range(0, constants.MY_TRIALS):
    dir_to_save = pjoin('dir_trial_' + str(t+1))
    os.mkdir(os.path.join(processed_dir, dir_to_save))

    for s in range(0 , constants.MY_SUBJECTS):
        for b in  range(0 , constants.BLOCKS):
            
            #fetching data
            raw = mne.io.RawArray( myTrials[t,s,b], info )
            
            #interpolation, picking channels
            raw.info['bads'] += ['T8', 'PZ']
            good_channels = mne.pick_types(raw.info, exclude='bads') 
            #raw.interpolate_bads(reset_bads=False)  #if the channels picks do not fail
            #raw.pick('')
            
            #filtering
            raw.notch_filter(np.arange(50, 125, 50))
            raw.filter(None, 90., fir_design='firwin')  # low pass filtering below 50 Hz
            raw.filter(7., 90.) 

            #alternative filter, not as good
            #mne.filter.filter_data(raw, sfreq=250, l_freq=7.5, h_freq=90)  
            
            #possible rereferencing
            #raw.set_eeg_reference([])  
            
            #resampling
            #raw_resampled = raw.copy().resample(125, npad='auto')  
            
            #ICA
        
            # saving to file for CNN to train on
            file_to_save = pjoin('subject_' + str(s+1) + '_block_' + str(b+1) + '.csv')
            full_path = os.path.join(processed_dir, dir_to_save, file_to_save)
            epoch = raw.get_data(picks = 'all')
            np.savetxt(full_path, epoch, delimiter=',')
        

databaseTrials = eeg_data_readers.wholeDatabaseReader()
info = mne.create_info(constants.DATABASE_CH_NAMES, constants.SAMPLING_FREQUENCY, ch_types='eeg')
            
processed_dir = 'database_processed_measurements'
if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
os.mkdir(processed_dir)

for t in range(0 , constants.DATABASE_TRIALS):
    dir_to_save = pjoin('dir_trial_' + str(t+1))
    os.mkdir(os.path.join(processed_dir, dir_to_save))

    for s in range(0 , constants.DATABASE_SUBJECTS):
        for b in  range(0 , constants.BLOCKS):
            
            #fetching data
            raw = mne.io.RawArray( databaseTrials[t,s,b], info )
            
            good_channels = mne.pick_types(raw.info, exclude='bads') 
            #raw.interpolate_bads(reset_bads=False)  #if the channels picks do not fail
            #raw.pick('')
            
            #filtering
            raw.notch_filter(np.arange(50, 125, 50))
            raw.filter(None, 90., fir_design='firwin')  # low pass filtering below 50 Hz
            raw.filter(7., 90.) 
            
            #alternative filter -> not as good
            #mne.filter.filter_data(raw, sfreq=250, l_freq=7.5, h_freq=90)  
            
            #possible rereferencing   -> everything is fine so also not used
            #raw.set_eeg_reference([])  
            
            #resampling  ->> no need performance okay
            #raw_resampled = raw.copy().resample(125, npad='auto')  
            
            #ICA
        
            # saving to file for CNN to train on
            file_to_save = pjoin('subject_' + str(s+1) + '_block_' + str(b+1) + '.csv')
            full_path = os.path.join(processed_dir, dir_to_save, file_to_save)
            epoch = raw.get_data(picks = 'all')
            np.savetxt(full_path, epoch, delimiter=',')
            


### some tryouts below -> not used

# artifact correction -> ICA

#raw_for_plot.filter(l_freq=1, h_freq=90) # for freq 1-90


            #raw_stft = mne.time_frequency.stft(mat_contents['results'][1], 4)
            
            #plt.plot(raw_stft, 1.0/1000 * np.abs(raw_stft))
            #plt.grid()
            #plt.show()
            
            #raw_ffty = fft(mat_contents['results'][1])
            #raw_fftx = fftfreq(1000,(1/250))
            #raw_fftx = fftshift(raw_fftx)
            
            #raw_fft = fftshift(raw_ffty)
            
            #plt.plot(raw_fftx, 1.0/1000 * np.abs(raw_fft))
            #plt.grid()
            #plt.show()


#epochs not tested
#epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,baseline=(None, 0), reject=reject, preload=False)  

#raw.resample(125, npad='auto')
#raw.plot_psd(fmin=2, fmax=62.5, average=True, spatial_colors=False);

#raw.info['bads'] += ['CP3', 'PZ'] 
#picks = mne.pick_types(raw.info, exclude='bads') 
#raw.plot()