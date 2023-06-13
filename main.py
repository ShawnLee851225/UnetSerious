
import numpy as np
import torch 
import torch.nn as nn
import argparse
import os
from torch.utils.data import DataLoader,Dataset
from Unet import UNet
from torchsummary import summary
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

"""----------module switch setting----------"""
tqdm_module = True #progress bar
torchsummary_module = True  #model Visual
argparse_module = True
load_model_weight = True
train_model  = False
"""----------module switch setting end----------"""
"""----------argparse module----------"""
if argparse_module:    
    parser = argparse.ArgumentParser(description = 'Object detection')
    parser.add_argument('--database_path',type=str,default='./dataset/',help='datapath')
    parser.add_argument('--modelpath',type=str,default='./model/',help='output model save path')
    parser.add_argument('--numpy_data_path',type=str,default='./numpydata/',help='output numpy data')
    parser.add_argument('--training_data_path',type=str,default='./training_process_data/',help='output training data path')

    parser.add_argument('--image_size',type=int,default= 540,help='image size')
    parser.add_argument('--num_classes',type=int,default= 1,help='num classes')
    parser.add_argument('--batch_size',type=int,default= 1,help='batch_size')
    parser.add_argument('--num_epoch',type=int,default= 100,help='num_epoch')
    parser.add_argument('--model',type= str,default='Unet',help='modelname')
    parser.add_argument('--optimizer',type= str,default='Adam',help='optimizer')
    parser.add_argument('--loss',type= str,default='CrossEntropyLoss',help='Loss')
    parser.add_argument('--lr',type= float,default=1e-3,help='learningrate')
    args = parser.parse_args()
"""----------argparse module end----------"""

"""----------function----------"""
class footplayerDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])

        # open image
        img = Image.open(img_path)
        label = Image.open(mask_path)

        # turn np
        img_np = np.array(img)
        label_np = np.array(label)
        # turn label 255 to 1
        label_np[label_np==255] = 1

        img = self.transforms(img_np)
        label = self.transforms(label_np)

        return img, label
    def __len__(self):
        return len(self.imgs)
train_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((args.image_size,args.image_size*16//9)),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),#做正規化[-1~1]之間
])

"""----------function end----------"""
"""----------tqdm init----------"""
if tqdm_module:
    pbar = tqdm(range(args.num_epoch),desc='Epoch',unit='epoch',maxinterval=1)
"""----------tqdm init end----------"""

train_set = footplayerDataset(args.database_path,train_transform)
train_loader = DataLoader(dataset = train_set,batch_size = args.batch_size,shuffle=False,pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = UNet(3,args.num_classes,True,False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,amsgrad=False)
loss = nn.BCEWithLogitsLoss()

if torchsummary_module:
    summary(model.to(device),(3,args.image_size,args.image_size*16//9))
for epoch in pbar:
    train_loss =0.0
    model.train()
    for images,label in train_loader:
        train_pred = model(images.to(device))
        batch_loss = loss(train_pred, label.to(device))

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item()
    print(f'train_loss:{train_loss}')
