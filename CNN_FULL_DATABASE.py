import os
import csv 
from pylab import rcParams
import collections 
import matplotlib.pyplot as plt
import numpy as np
import mne
import torch
import torch.nn as nn
import torchvision.transforms
from sklearn.preprocessing import RobustScaler
import scipy.io

from os.path import dirname, join as pjoin

import constants
import eeg_data_readers

#initiazization of pytorch and scaler
torch.manual_seed(100)
scaler = RobustScaler()


#initialization of CNN parameters
eeg_sample_length = 64000
hidden_layer_1 = 700 
hidden_layer_2 = 1300 
hidden_layer_3 = 150 
output_layer = 10 
number_of_classes = 1 
learning_rate = 2e-5

results_positive_passed = np.zeros(constants.MY_TRIALS)
results_positive_failed = np.zeros(constants.MY_TRIALS)
results_negative_passed = np.zeros(constants.MY_TRIALS)
results_negative_failed = np.zeros(constants.MY_TRIALS)
results_1 = np.zeros(constants.MY_TRIALS)
results_2 = np.zeros(constants.MY_TRIALS)

myTrials = eeg_data_readers.myMeasurements_fif_reader('my_processed_measurements')
dataBaseTrials = eeg_data_readers.databaseMeasurements_fif_reader('database_processed_measurements')


for current_trial in range (0, constants.DATABASE_TRIALS):
    
    epochsOK = []
    for s in range(0 , constants.DATABASE_SUBJECTS):
        for b in  range(0 , constants.BLOCKS):
            
            epochsOKb4 = (dataBaseTrials[current_trial,s,b])
            epochsOKb4 = epochsOKb4.transpose()
            epochsOKb4 = scaler.fit_transform(epochsOKb4)
            epochsOKb4 = epochsOKb4.transpose()
            
            epochsOKb4 = np.concatenate(epochsOKb4)
            epochsOK.append(epochsOKb4)
    
    epochsBAD = []
    for s in range(0 , constants.DATABASE_SUBJECTS):
        for b in  range(0 , constants.BLOCKS):
            for t in range (0, constants.DATABASE_TRIALS):
                if t != current_trial:
                    epochsBADb4 = dataBaseTrials[current_trial,s,b]
                    
                    # Scaling of the data
                    epochsBADb4 = np.array(epochsBADb4)
                    epochsBADb4 = epochsBADb4.transpose()
                    epochsBADb4 = scaler.fit_transform(epochsBADb4)
                    epochsBADb4 = epochsBADb4.transpose()
                    
                    epochsBADb4 = np.concatenate(epochsBADb4)
                    epochsBAD.append(epochsBADb4)
                    
                    epochsBADb4 = dataBaseTrials[t,s,b]

            
    np.squeeze(epochsOK)
    np.squeeze(epochsBAD)
    
    
    
    # Preparation of all the tesors for both training and testing
    epochsOK_train = epochsOK[0:240]
    epochsOK_test = epochsOK[240:280]
    epochsOK_test = torch.tensor(epochsOK_test).float()
    
    epochsBAD_train = epochsBAD[1200:1600]
    epochsBAD_test = epochsBAD[1600:1800]
    epochsBAD_test = torch.tensor(epochsBAD_test).float()
    
    training_data = torch.tensor(np.concatenate((epochsOK_train, epochsBAD_train), axis = 0)).float()
    positive_testing_data = torch.tensor(epochsOK_test).float()
    negative_testing_data = torch.tensor(epochsBAD_test).float()
    
    print("shape of the training data " + str(training_data.shape))
    print("shape of good trials " + str(positive_testing_data.shape))
    print("shape of bad trials " + str(negative_testing_data.shape))
    
    #labeling of the data
    labels = torch.tensor(np.zeros((training_data.shape[0],1))).float()
    labels[0:240] = 1.0
    print("shape of trainig labes: " + str(labels.shape))
    
    
    #creation of the whole CNN
    CNN_model = nn.Sequential()
    
    # Input Layer
    CNN_model.add_module('Input Linear', nn.Linear(eeg_sample_length, hidden_layer_1))
    CNN_model.add_module('Input Activation', nn.CELU()) 
    
    # Layer 1
    CNN_model.add_module('Hidden Linear', nn.Linear(hidden_layer_1, hidden_layer_2))
    CNN_model.add_module('Hidden Activation', nn.ReLU())
    
    # Layer 2
    CNN_model.add_module('Hidden Linear2', nn.Linear(hidden_layer_2, hidden_layer_3))
    CNN_model.add_module('Hidden Activation2', nn.ReLU())
    
    # Layer 3
    CNN_model.add_module('Hidden Linear3', nn.Linear(hidden_layer_3, output_layer))
    CNN_model.add_module('Hidden Activation3', nn.ReLU())
    
    # Output Layer
    CNN_model.add_module('Output Linear', nn.Linear(output_layer, number_of_classes))
    CNN_model.add_module('Output Activation', nn.Sigmoid())
    
    # Loss function for the learning curve
    loss_function = torch.nn.MSELoss()
    
        # Define a training procedure
    def CNN_training(train_data, actual_class, iterations):
    
        # Keep track of loss at every training iteration
        loss_data = []
    
        # Training iteration with specified learning curve
        for i in range(iterations):
            
            # classification model
            class_model = CNN_model(train_data)
            
            # Find out how wrong the network was
            loss = loss_function(class_model, actual_class)
            loss_data.append(loss)
    
            # optimizer gradient
            optimizer.zero_grad()
    
            # feedback
            loss.backward()
            optimizer.step()
    
    
    
    # Saving and loading of the default state of the CNN
    CNN_name = pjoin("CNN_default_state_" + str(current_trial))
    torch.save(CNN_model, CNN_name)
    CNN_model = torch.load(CNN_name)
    
    # Definition of the learning f-n
    optimizer = torch.optim.Adam(CNN_model.parameters(), lr = learning_rate)
    
    # Training of the data
    CNN_training(training_data, labels, iterations = 100)
    
    positive_passed = 0
    positive_failed = 0
    negative_passed = 0
    negative_failed = 0
    
    
    # Classify our positive test dataset and print the results
    classification_good = CNN_model(positive_testing_data)
    for index, value in enumerate(classification_good.data.tolist()):
        if(value[0] > 0.5):
            positive_passed = positive_passed + 1
        else:
            positive_failed = positive_failed + 1

    
    # Classify our negative test dataset and print the results
    classification_bad = CNN_model(negative_testing_data)
    for index, value in enumerate(classification_bad.data.tolist()):
        if(value[0] < 0.5):
            negative_passed = negative_passed + 1
        else:
            negative_failed = negative_failed + 1
            
    final_result_1 = (positive_passed) / (positive_passed + positive_failed) * 100
    final_result_2 = (negative_passed) / (negative_passed + negative_failed) * 100
    
    results_positive_passed[current_trial] = positive_passed
    results_positive_failed[current_trial] = positive_failed
    results_negative_passed[current_trial] = negative_passed
    results_negative_failed[current_trial] = negative_failed
    
    results_1[current_trial] = final_result_1
    results_2[current_trial] = final_result_2
    
    print("stats positive: " + str(positive_passed) + "/" + str(positive_failed)) 
    print("stats negative: " + str(negative_passed) + "/" + str(negative_failed))

      

results_dir = "results/database"
os.mkdir(results_dir)

path = pjoin(results_dir, "results_positive_classification_whole")
np.save(path,results_1)

path = pjoin(results_dir, "results_negative_classification_whole")
np.save(path,results_2)

path = pjoin(results_dir, "results_positive_passed_whole")
np.save(path,results_positive_passed)

path = pjoin(results_dir, "results_positive_failed_whole")
np.save(path,results_positive_failed)

path = pjoin(results_dir, "results_negative_passed_whole")
np.save(path,results_negative_passed)

path = pjoin(results_dir, "results_negative_failed_whole")
np.save(path,results_negative_failed)