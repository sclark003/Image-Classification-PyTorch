import pickle
import time
import os
import re

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# function to train model
def training(model, train_dl, num_epochs, lr, path):
    # tracking variables
    train_hist = {}
    train_hist['Losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    train_hist['Accuracy'] = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # Set device
    criterion = nn.CrossEntropyLoss().to(device)                               # Binary Cross Entropy Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)                    # Set optimiser

    start_time = time.time()                                                   # start time
    
    # training loop
    for epoch in range(num_epochs):
        model.train()
        Loss = []
        predictions_list = np.array([])
        labels_list = np.array([])
        epoch_start_time = time.time()
        
        for (image, labels) in tqdm(train_dl):
             image = image.to(device)
             labels = labels.to(device)

             # --------train the network---------- #
             outputs = model(image)
             _, predicted = torch.max(outputs,1)                               # get class predictions           
             
             # Zero the parameter gradients
             optimizer.zero_grad()
             
             # compute the loss for real and fake images
             loss = criterion(outputs, labels)
             loss.backward()
             optimizer.step()
             
             # Keep stats for Loss and Accuracy
             Loss.append(loss.item())
             predictions_list = np.append(predictions_list,predicted.detach().cpu().numpy())
             labels_list = np.append(labels_list,labels.detach().cpu().numpy())
    
        # Calculate epoch stats
        epoch_loss = np.mean(Loss)
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        accuracy = f1_score(labels_list, predictions_list,average='micro')     # F1 Accuracy score
             
        # Display epoch stats
        print("Epoch %d of %d with %.2f s" % (epoch + 1, num_epochs, per_epoch_ptime))
        print("Loss: %.8f" % (epoch_loss))
        print("Accuracy: %.8f" % (accuracy))
        
        # Save epoch stats
        train_hist['Losses'].append(epoch_loss)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        train_hist['Accuracy'].append(accuracy)
        
        # Save checkpoint
        torch.save({
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': epoch_loss,
             'train_hist': train_hist
             }, path+"check/"+"_"+str(epoch+1)+".pt")
             #}, "/content/drive/MyDrive/check/checkpoint"+"_"+str(epoch+1)+".pt")

    # Stop timing
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)
    
    print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), num_epochs, total_ptime))
    print("Training finish!... save training results")
    with open(path + 'train_hist.pkl', 'wb') as f:
    #with open(path + '/content/drive/MyDrive/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)


# function to test model on validation set
def inference (model, val_dl):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # Set device    
    predictions_list = np.array([])
    labels_list = np.array([])
    topOne = 0.0
    topFive = 0.0
    i = 0.0

    # Disable gradient updates
    with torch.no_grad():
        for (image, labels) in tqdm(val_dl):
            image = image.to(device)
            labels = labels.to(device)

            # --------test the network---------- #
            outputs = model(image)
            _, predicted = torch.max(outputs,1)                               # get class predictions           
           #print(predicted)
            topOne, topFive = topFivePredict(outputs.detach().cpu().numpy(),labels.detach().cpu().numpy(),topOne,topFive)
           #print("\n",topOne, topFive)
            
            # Keep stats for Accuracy
            predictions_list = np.append(predictions_list,predicted.detach().cpu().numpy())
            labels_list = np.append(labels_list,labels.detach().cpu().numpy())
            i += 1
            
    
    print("A:", f1_score(labels_list, predictions_list, average='micro'))      # F1 Accuracy score
    topOneAccuracy = topOne/(len(val_dl)*i)                                    # Top-1 Accuracy
    topFiveAccuracy = topFive/(len(val_dl)*i)                                  # Top-5 Accuracy
    print("Top One Accuracy:", topOneAccuracy, ", Top Five Accuracy:", topFiveAccuracy)
    conf = confusion_matrix(labels_list, predictions_list)                     # Calculate confusion matrix
    return conf                                                                # return confusion matrix


# function to calculate top-5 and top-1 accuracies
def topFivePredict(outputs,labels,xOne, xFive):
    # Accuracy = No. correct/ Total No.
    for i in range(len(outputs)):        # loop through batch outputs
        x = outputs[i]
        sorted_x = np.argsort(x)         # sort by location according to size
        sortedFive = sorted_x[-5:]       # select five highest locations (i.e. predictions)
        sortedOne = sorted_x[-1]         # select highest location (i.e. prediction)
        
        if labels[i] in sortedFive:      # if true label is in top 5
            xFive += 1                   # increase correct predictions
        if labels [i] == sortedOne:      # if true label is top prediction
            xOne += 1                    # increase correct predictions
    
    return xOne, xFive                   # return number of correct predictions
    

# function to get train data for Tiny-ImageNet dataset
def getTrainData(path):
    path = path+"train/"
    #path = "/content/drive/MyDrive/train/"
    meta_data = []
    for entry in os.scandir(path):                                   # for each class in train data
        class_idx = entry.name                                       # save class name
        relative_path = path+entry.name+"/images/"                   # save relative path to data
        for image in os.scandir(relative_path):                      # for image in class folder
            meta_data.append((relative_path+image.name,class_idx))   # save relative path and class name
    return meta_data                                                 # return list of data items


# function to get val data for Tiny-ImageNet dataset
def getValData(path):
    meta_data = []
    path = path+"val/"
    #path = "/content/drive/MyDrive/val/"
    f = open(path+"val_annotations.txt", "r")         # open txt file with class data for validation set
    for line in f:                                    # read each line
        line = re.split(r'\t+', line)                 # split line by tabs
        image = path+"images/"+line[0]                # save relative path to each data item
        class_idx = line[1]                           # save class name
        meta_data.append((image,class_idx))           # save relative path and class name
    return meta_data                                  # return list of data items
