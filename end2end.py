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
	device = torch.device("cuda")
else:
	device = torch.device("cpu")


#################################################################
######################### Load test data ########################
test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0,1,10]),
                                           transform=transforms.Compose([
                                               Rescale((64,64)),
                                               ToTensor(),
                                           ]))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=1)

unresized_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=None)



#################################################################
######################### Load models ###########################

model = pickle.load(open('saved-model.pth', 'rb'))
model = model.to(device)
test_loss = evaluate(model, test_loader, criterion)
LOG_INFO('test loss (localizer) = %.4f' % (test_loss))


names = ['eyebrow', 'eye', 'nose', 'mouth']
models={}
for name in names:
  models[name] = pickle.load(open('saved-model-%s.pth'%name, 'rb'))


#################################################################
############### Calculate centroid and extract_parts ############

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


def extract_parts(indexs, centroids, orig_dataset):
  orig_images = torch.Tensor([])
  orig_labels = torch.Tensor([])
  res = {}

  for i,idx in enumerate(indexs):

    l,h,w = orig_dataset[idx]['labels'].shape
    offset_y, offset_x = (box_size-h)//2, (box_size-w)//2

    image = torch.zeros(3,box_size,box_size)
    labels = torch.zeros(l,box_size,box_size)
    image[:,offset_y:offset_y+h, offset_x:offset_x+w] = orig_dataset[idx]['image']
    labels[:,offset_y:offset_y+h, offset_x:offset_x+w] = orig_dataset[idx]['labels']

    orig_images = torch.cat([orig_images, image])
    orig_labels = torch.cat([orig_labels, labels])

    # Scale and shift centroids
    centroids[i] =  centroids[i] * torch.Tensor([h/64., w/64.]).view(1,2).to(device) \
                                 + torch.Tensor([offset_y, offset_x]).view(1,2).to(device)


  orig_images = orig_images.to(device).view(len(indexs),3,box_size,box_size)
  orig_labels = orig_labels.to(device).view(len(indexs),l,box_size,box_size)

  #################
  # Non-Mouth parts
  index = centroids.index_select(1, torch.tensor(range(5)).to(device)).long()
  n_parts = index.shape[-2]

  # Construct repeated image of n x p x c x h x w
  repeated_images = orig_images.unsqueeze(1).repeat_interleave(n_parts, dim=1)
  repeated_labels = orig_labels.unsqueeze(1).repeat_interleave(n_parts, dim=1)

  # Calculate index of patches of the form n x p x 64 x 64 corresponding to each facial part
  # After this index_x/y will be n x p x 64 x 64
  index_y = index[:,:,0].unsqueeze(-1) + torch.from_numpy(np.arange(-32,32)).view(1,1,64).to(device)
  index_y = index_y.unsqueeze(-1).repeat_interleave(box_size, dim=-1)

  index_x = index[:,:,1].unsqueeze(-1) + torch.from_numpy(np.arange(-32,32)).view(1,1,64).to(device)
  index_x = index_x.unsqueeze(-2).repeat_interleave(64, dim=-2)

  # Get patch images (n x p x c x h x w)
  patch_images = torch.gather(repeated_images, -2, index_y.unsqueeze(2).repeat_interleave(3,dim=2) )
  patch_images = torch.gather(patch_images, -1, index_x.unsqueeze(2).repeat_interleave(3,dim=2) )

  # Get patch labels (n x p x l x h x w)
  patch_labels = torch.gather(repeated_labels, -2, index_y.unsqueeze(2).repeat_interleave(l,dim=2) )
  patch_labels = torch.gather(patch_labels, -1, index_x.unsqueeze(2).repeat_interleave(l,dim=2) )

  res['mouth'] = {'patch_images': patch_images, 'labels': patch_labels}


  
  ##################
  # Mouth part
  index = centroids.index_select(1, torch.tensor(range(5,8)).to(device)).mean(dim=1, keepdim=True).long()

  # Construct repeated image of n x 1 x c x h x w
  repeated_images = orig_images.unsqueeze(1)
  repeated_labels = orig_labels.unsqueeze(1)
  
  # Calculate index of mouth patches of the form n x 1 x 80 x 80 corresponding mouth part
  # After this index_x/y will be n x 1 x 80 x 80
  index_y = index[:,:,0].unsqueeze(-1) + torch.from_numpy(np.arange(-40,40)).view(1,1,80).to(device)
  index_y = index_y.unsqueeze(-1).repeat_interleave(box_size, dim=-1)

  index_x = index[:,:,1].unsqueeze(-1) + torch.from_numpy(np.arange(-40,40)).view(1,1,80).to(device)
  index_x = index_x.unsqueeze(-2).repeat_interleave(80, dim=-2)

  # Get patch images (n x 1 x c x 80 x 80)
  patch_images = torch.gather(repeated_images, -2, index_y.unsqueeze(2).repeat_interleave(3,dim=2) )
  patch_images = torch.gather(patch_images, -1, index_x.unsqueeze(2).repeat_interleave(3,dim=2) )

  # Get patch labels (n x 1 x l x 80 x 80)
  patch_labels = torch.gather(repeated_labels, -2, index_y.unsqueeze(2).repeat_interleave(l,dim=2) )
  patch_labels = torch.gather(patch_labels, -1, index_x.unsqueeze(2).repeat_interleave(l,dim=2) )

  res['non-mouth'] = {'patch_images': patch_images, 'labels': patch_labels}

  return res



with torch.no_grad():
  for batch in loader:
    images, labels, indexs = batch['labels'].to(device), batch['index']
    pred_labels = F.softmax(model(images), 1)
    centroids = calculate_centroids(pred_labels)

    res = extract_parts(indexs, centroids, orig_dataset)

    # Prepare batches for facial parts
    batchs={}
    batchs['eyebrow'] =
    batchs['eye'] =
    batchs['nose'] =
    batchs['mouth'] =

    # Get prediction

    # Rearrange patch results onto original image

    # Save mask result