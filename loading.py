import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import cv2
from flask import Flask,jsonify,request,make_response,url_for,redirect
import requests, json
app = Flask(__name__)
from io import BytesIO
import base64


data_dir  = 'Garbage classification\Garbage classification'

classes = os.listdir(data_dir)
# print(classes)

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

dataset = ImageFolder(data_dir, transform = transformations)

train_ds, val_ds, test_ds = random_split(dataset, [1804,173 ,550 ])
# print(len(train_ds), len(val_ds), len(test_ds))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
 
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
        

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        #using the resnet model available already
        self.network = models.resnet50(pretrained=False)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

model = ResNet()
# batch_size=32
# train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
# val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

device = get_default_device()
# print(device)

# train_dl = DeviceDataLoader(train_dl, device)
# val_dl = DeviceDataLoader(val_dl, device)
model = ResNet()

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

model = torch.load('BTP_model.pt',map_location=torch.device('cpu'))
# print(model.eval())

# for i in range(100):
#     print()

# img = Image.open('g.jpeg')

# nimg = transformations(img)
# print('Predicted:', predict_image(nimg,model))
# img.show()
# while(True):
#     i = input('Another Image?(Y/N):')
#     if i in ['Y','y']:
#         cam = cv2.VideoCapture(1)
#         res,img = cam.read()
#         if res:
#             # cv2.imshow('Image for Class',img)
#             # cv2.waitKey(0)
#             img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(img)
            
#             # img = Image.open('g.jpeg')
#             nimg = transformations(img)
#             print('Predicted:', predict_image(nimg, model))
#             img.show()
#         else:
#             print('No Image Found')
#     else:
#         break

def predict(img):
    nimg = transformations(img)
    return {"Type": predict_image(nimg, model)}
@app.route('/predict', methods=['GET','POST'])
def create_row_in_gs():
    if request.method == 'GET':
        return make_response('failure')
    if request.method == 'POST':
        img = request.json['img']
        print(img)
        dataBytesIO = BytesIO(base64.b64decode(img))
        dataBytesIO.seek(0)
        im = Image.open(dataBytesIO)
        response = requests.post(
            data=json.dumps(predict(im)),
            headers={'Content-Type': 'application/json'}
        )
        return response.content

if __name__ == '__main__':
    app.run(host='localhost',debug=True, use_reloader=True)