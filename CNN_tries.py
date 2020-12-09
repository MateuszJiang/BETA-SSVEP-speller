from collections import OrderedDict
from pylab import rcParams
import torch
import torch.nn as nn
import torchvision.transforms
import matplotlib.pyplot as plt
import numpy as np
import mne
from sklearn.preprocessing import RobustScaler
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
import csv 

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

results = np.zeros(constants.DATABASE_TRIALS)

myTrials = eeg_data_readers.myMeasurements_fif_reader('my_processed_measurements')
dataBaseTrials = eeg_data_readers.databaseMeasurements_fif_reader('database_processed_measurements')

#division into epochs for training
epochsOK = []
for b in  range(0 , constants.BLOCKS):
    for s in range(0 , constants.DATABASE_SUBJECTS):
        epochsOKb4 = (dataBaseTrials[1,s,b])
        
        #Scaling of the data
        epochsOKb4 = epochsOKb4.transpose()
        epochsOKb4 = scaler.fit_transform(epochsOKb4)
        epochsOKb4 = epochsOKb4.transpose()
        
        
        epochsOKb4 = np.concatenate(epochsOKb4)
        epochsOK.append(epochsOKb4)
        
epochsBAD = []
for t in range (0, constants.DATABASE_TRIALS):
    for s in range(0 , constants.DATABASE_SUBJECTS):
        for b in  range(0 , constants.BLOCKS):
                if t != 1:
                    epochsBADb4 = dataBaseTrials[t,s,b]

                    # Scaling of the data
                    epochsBADb4 = np.array(epochsBADb4)
                    epochsBADb4 = epochsBADb4.transpose()
                    epochsBADb4 = scaler.fit_transform(epochsBADb4)
                    epochsBADb4 = epochsBADb4.transpose()
                    
                    epochsBADb4 = np.concatenate(epochsBADb4)
                    epochsBAD.append(epochsBADb4)
                    
np.squeeze(epochsOK)
np.squeeze(epochsBAD)

# Preparation of all the tesors for both training and testing
epochsOK_train = epochsOK[0:240]
epochsOK_test = epochsOK[240:280]
epochsOK_test = torch.tensor(epochsOK_test).float()
    
epochsBAD_train = epochsBAD[200:1600]
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
def train_network(train_data, actual_class, iterations):

    # Keep track of loss at every training iteration
    loss_data = []

    # Begin training for a certain amount of iterations
    for i in range(iterations):
        
        # Begin with a classification
        classification = tutorial_model(train_data)
        
        print("training data part: " + str(i))
        
        # Find out how wrong the network was
        loss = loss_function(classification, actual_class)
        loss_data.append(loss)

        # Zero out the optimizer gradients every iteration
        optimizer.zero_grad()

        # Teach the network how to do better next time
        loss.backward()
        optimizer.step()
  
    # Plot a nice loss graph at the end of training
    rcParams['figure.figsize'] = 10, 5
    plt.title("Loss function")
    plt.plot(list(range(0, len(loss_data))), loss_data)
    plt.show()

# Saving and loading of tge default state of the CNN
torch.save(CNN_model, "CNN_default_state")
tutorial_model = torch.load("CNN_default_state")

# Definition of the learning f-n
optimizer = torch.optim.Adam(tutorial_model.parameters(), lr = learning_rate)

# Training of the data
train_network(training_data, labels, iterations = 100)

positive_passed = 0
positive_failed = 0
negative_passed = 0
negative_failed = 0

# Classify our positive test dataset and print the results
classification_1 = tutorial_model(positive_testing_data)
for index, value in enumerate(classification_1.data.tolist()):
    print("Positive Classification {1}: {0:.2f}%".format(value[0] * 100, index + 1))
    if(value[0] > 0.35):
        positive_passed = positive_passed + 1
    else:
        positive_failed = positive_failed + 1

print()

# Classify our negative test dataset and print the results
classification_2 = tutorial_model(negative_testing_data)
for index, value in enumerate(classification_2.data.tolist()):
    print("Negative Classification {1}: {0:.2f}%".format(value[0] * 100, index + 1))
    if(value[0] < 0.35):
        negative_passed = negative_passed + 1
    else:
        negative_failed = negative_failed + 1
        
final_result = (positive_passed + negative_passed) / (positive_passed + negative_passed + positive_failed + negative_failed) * 100

print("stats positive: " + str(positive_passed) + "/" + str(positive_failed)) 
print("stats negative: " + str(negative_passed) + "/" + str(negative_failed))
print("Final results: " + str(final_result))
