from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# add output layer to models
class Net(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model               
        self.fc1 = nn.Linear(1000, 10)   
    
    def forward(self, x): # input = [16, 1, 64, 2579]
        x = self.model(x) # load pytorch model
        x = self.fc1(x)   # output 10 class predictions
        return x
  
    
# add output layer to inception model
class NetLogits(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fc1 = nn.Linear(1000, 10)
    
    def forward(self, x): # input = [16, 1, 64, 2579]
        x = self.model(x) # load pytorch model
        x = x.logits      # take logits output
        x = self.fc1(x)   # output 10 class predictions
        return x    


# add output layer to models
class Net2(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fc1 = nn.Linear(1000, 200)
    
    def forward(self, x): # input = [16, 1, 64, 2579]
        x = self.model(x) # load pytorch model
        x = self.fc1(x)   # output 200 class predictions
        return x
    
    
# add output layer to inception model
class NetLogits2(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fc1 = nn.Linear(1000, 200)
    
    def forward(self, x): # input = [16, 1, 64, 2579]
        x = self.model(x) # load pytorch model
        x = x.logits      # take logits output
        x = self.fc1(x)   # output 200 class predictions
        return x    


# dataset object for Tiny-ImageNet data
class DSLoader(Dataset):
    def __init__(self, labels,n,class_ids):
        self.labels = labels
        self.n = n
        self.class_ids = class_ids
    
            
    def __len__(self):
        return len(self.labels)    
        

    def __getitem__(self, idx):
        
        # preprocess data
        preprocess = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(self.n),
            transforms.ToTensor(),                         
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        image_path = self.labels[idx][0]             # relative path to image data
        class_name = self.labels[idx][1]             # true class of image data
        class_id = self.class_ids.index(class_name)  # true class numeric id
        image = Image.open(image_path)               # load image
        image = preprocess(image)                    # preprocess image
        
        return image, class_id                       # return image data and true class label