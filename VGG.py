import numpy as np
import Model
import Train_Eval as te
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision import transforms
from sklearn.metrics import ConfusionMatrixDisplay


# initialise the device for training, if gpu is available, device = 'cuda', else: device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training parameters
batch_size = 100
learning_rate = 0.0002
epochs = 6

# -----------------MNIST-----------------#
data_dir = './MNIST_data/'
save_dir = './MNIST_results/vgg/'
#data_dir = '/content/drive/MyDrive/MNIST_data/'
#save_dir = '/content/drive/MyDrive/MNIST_results/vgg/'

# preprocess data
preprocess = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),                            
])

# construct the dataset and data loader
train_data = datasets.MNIST(root=data_dir, train=True, transform=preprocess, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_data = datasets.MNIST(root=data_dir, download=True, transform=preprocess, train=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
# ---------------------------------------#

# # -----------------CIFAR-----------------#
# data_dir = './CIFAR_data/'
# save_dir = './CIFAR_results/vgg/'
# #data_dir = '/content/drive/MyDrive/CIFAR_data/'
# #save_dir = '/content/drive/MyDrive/CIFAR_results/vgg/'

# # preprocess data
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),                         
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# # construct the dataset and data loader
# train_data = data = datasets.CIFAR10(root=data_dir, train=True, transform=preprocess, download=True)
# train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# val_data = datasets.CIFAR10(root=data_dir, download=True, transform=preprocess, train=False)
# val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
# # ---------------------------------------#

# #---------------ImageNet----------------#
# data_dir = './ImageNet_data/'
# save_dir = './ImageNet_results/vgg/'
# # data_dir = '/content/drive/MyDrive/ImageNet_data/'
# # save_dir = '/content/drive/MyDrive/ImageNet_results/vgg/'

# # get list of class ids 
# f = open(data_dir+"wnids.txt", "r")
# class_ids = []                                                                                                                              
# for line in f:
#     class_ids.append(line[0:9])


# # construct the dataset and data loader
# train_data = te.getTrainData(data_dir)
# train_ds = Model.DSLoader(train_data,224,class_ids)
# train_loader = DataLoader(train_ds, batch_size, shuffle=True)

# val_data = te.getValData(data_dir)
# val_ds = Model.DSLoader(val_data,224,class_ids)
# val_loader = DataLoader(val_ds, batch_size, shuffle=False)
# #---------------------------------------#

# model
v = models.vgg16(pretrained=False)
model = Model.Net(v)    # 10 classes
#model = Model.Net2(v)  # 200 classes
model = model.to(device)
#print(model)

# Train
te.training(model, train_loader, epochs, learning_rate, save_dir)

###############################################################################
# # load checkpoint
# checkpoint = torch.load(save_dir+"check/checkpoint_6.pt",map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# Run inference on trained model with the validation set
cm = te.inference(model, val_loader)
# diagonal = cm.diagonal()/cm.sum(axis=1)  # see class accuracies
# print(diagonal)

# plot confusion matrix
cm2 = cm / cm.astype(np.float).sum(axis=1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=range(0,10))
disp = disp.plot()




