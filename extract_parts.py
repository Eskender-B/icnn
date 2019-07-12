import torch
import torch.nn as nn
from model import ICNN
from utils import LOG_INFO
from preprocess import Rescale, ToTensor, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import argparse
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
import shutil
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=25, type=int, help="Batch size")
args = parser.parse_args()
print(args)


if torch.cuda.is_available():
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

train_dataset = ImageDataset(txt_file='exemplars.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=transforms.Compose([
                                               Rescale((64,64)),
                                               ToTensor()
                                           ]))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


valid_dataset = ImageDataset(txt_file='tuning.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=transforms.Compose([
                                               Rescale((64,64)),
                                               ToTensor()
                                           ]))
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=transforms.Compose([
                                               Rescale((64,64)),
                                               ToTensor(),
                                           ]))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

unresized_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=None)


def calculate_centroids(tensor):
  tensor = tensor.float() + 1e-10
  n,l,h,w = tensor.shape
  indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
  indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
  center_y = tensor.sum(3) * indexs_y.view(1,1,-1) 
  center_y = center_y.sum(2, keepdim=True) / tensor.sum([2,3]).view(n,l,1)
  center_x = tensor.sum(2) * indexs_x.view(1,1,-1)
  center_x = center_x.sum(2, keepdim=True) / tensor.sum([2,3]).view(n,l,1)
  return torch.cat([center_y, center_x], 2)


def extract_parts(loader, orig_dataset):
  root_dir='data/facial_parts'
  with torch.no_grad():
    for batch in loader:
      labels, indexs = batch['labels'].to(device), batch['index']
      labels = F.normalize(labels.to(device), 1)
      centroids = calculate_centroids(labels)
      
      orig_images = torch.Tensor([]).to(device)
      orig_labels = torch.Tensor([]).to(device)

      for i in indexs:

        l,h,w = orig_dataset[i]['labels'].shape
        offset_y, offset_x = (512-h)//2, (512-w)//2

        image = torch.zeros(3,512,512)
        labels = torch.zeros(l,512,512)
        image[:,offset_y:offset_y+h, offset_x:offset_x+w] = orig_dataset[i]['image']
        labels[:,offset_y:offset_y+h, offset_x:offset_x+w] = orig_dataset[i]['labels']

        orig_images = torch.cat([orig_images, image])
        orig_labels = torch.cat([oirg_labels, labels])

        # Scale and shift centroids
        centroids[i] =  centroids[i] * torch.Tensor([h/64., w/64.]).view(1,2).to(device) \
                                     + torch.Tensor(offset_y, offset_x).view(1,2).to(device)

      orig_images = orig_images.view(len(indexs),3,512,512)
      orig_labels = orig_labels.view(len(indexs),l,512,512)

      non_mouth_index = centroids.index_select(1, range(6)).long()
      non_mouth_index_y = non_mouth_index[:,:,0] + torch.from_numpy(np.arange(-32,32)).view(1,1,64).to(device)
      non_mouth_index_x = non_mouth_index[:,:,1] + torch.from_numpy(np.arange(-32,32)).view(1,1,64).to(device)

      #n x p x c x h x w
      index_y = torch.repeat_interleave(non_mouth_index_y.unsqueeze(2),3, dim=2)
      index_x = torch.repeat_interleave(non_mouth_index_x.unsqueeze(2),3, dim=2)

      #n x p x c x 64 x 64
      non_mouth_patches = orig_images.gather(2, )
      non_mouth_labels = 

      mouth_index = centroids.index_select(1, range(6,9)).mean(dim=1).long()
      mouth_patches =
      mouth_labels =  


      #### Eyebrow
      # Left

      # Right
      # Eye

      # Nose

      # Mouth