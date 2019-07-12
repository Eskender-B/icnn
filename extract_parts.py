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
from skimage import io

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


unresized_train = ImageDataset(txt_file='exemplars.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=None)
unresized_valid = ImageDataset(txt_file='tuning.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=None)
unresized_test = ImageDataset(txt_file='testing.txt',
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

def save_patches(image, labels, dataset, indexs):
  n,p,c,w,h = image.shape
  _,_,l,_,_ = labels.shape
  name_list = dataset.name_list

  image = image.to('cpu').numpy().transpose([4,3,1,0,2])
  labels = labels.to('cpu').numpy()

  for i in range(n):
    # Save each patch in an image
    if p ==1:
      #mouth
      image_name = shutil.os.path.join(root_dir, dic[5], 'images', name_list[indexs[i], 1].strip() + '.jpg')
      io.imsave(image_name, image[i][0])

      shutil.os.mkdir(shutil.os.path.join(root_dir, dic[5], 'labels', name_list[indexs[i], 1].strip()))
      label_name = shutil.os.path.join(root_dir, dic[5], 'labels', name_list[indexs[i], 1].strip(), name_list[indexs[i], 1].strip() + '_lbl%.2d.png')
      for k in range(l):
        io.imsave(label_name%k, labels[i][0][k])

    else:
      #non-mouth
      for pp in range(p):
        image_name = shutil.os.path.join(root_dir, dic[pp], 'images', name_list[indexs[i], 1].strip() + '.jpg')
        io.imsave(image_name, image[i][0])

        shutil.os.mkdir(shutil.os.path.join(root_dir, dic[pp], 'labels', name_list[indexs[i], 1].strip()))
        label_name = shutil.os.path.join(root_dir, dic[pp], 'labels', name_list[indexs[i], 1].strip(), name_list[indexs[i], 1].strip() + '_lbl%.2d.png')
        for k in range(l):
          io.imsave(label_name%k, labels[i][pp][k])




def extract_parts(loader, orig_dataset):
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


      orig_images = orig_images.to(device).view(len(indexs),3,512,512)
      orig_labels = orig_labels.to(device).view(len(indexs),l,512,512)

      #################
      # Non-Mouth parts
      index = centroids.index_select(1, range(5)).long()
      n_parts = index.shape[-2]

      # Construct repeated image of n x p x c x h x w
      repeated_images = orig_images.unsqueeze(1).repeat_interleave(n_parts, dim=1)
      repeated_labels = orig_labels.unsqueeze(1).repeat_interleave(n_parts, dim=1)

      # Calculate index of patches of the form n x p x h x w corresponding to each facial part
      # After this index_x/y will be n x p x h x w
      index_y = index[:,:,0] + torch.from_numpy(np.arange(-32,32)).view(1,1,64).to(device)
      index_y = index_y.unsqueeze(-1).repeat_interleave(512, dim=-1)

      index_x = index[:,:,1] + torch.from_numpy(np.arange(-32,32)).view(1,1,64).to(device)
      index_x = index_x.unsqueeze(-2).repeat_interleave(64, dim=-2)

      # Get patch images (n x p x c x h x w)
      patch_images = torch.gather(repeated_images, dim=-2, index_y.unsqueeze(2).repeat_interleave(3,dim=2) )
      patch_images = torch.gather(patch_images, dim=-1, index_x.unsqueeze(2).repeat_interleave(3,dim=2) )

      # Get patch labels (n x p x l x h x w)
      patch_labels = torch.gather(repeated_labels, dim=-2, index_y.unsqueeze(2).repeat_interleave(l,dim=2) )
      patch_labels = torch.gather(patch_labels, dim=-1, index_x.unsqueeze(2).repeat_interleave(l,dim=2) )


      # Save patches
      save_patches(patch_images, patch_labels, orig_dataset, indexs)

      
      ##################
      # Mouth part
      index = centroids.index_select(1, range(5,8)).mean(dim=1, keepdim=True).long()

      # Construct repeated image of n x 1 x c x h x w
      repeated_images = orig_images.unsqueeze(1)
      repeated_labels = orig_labels.unsqueeze(1)
      
      # Calculate index of mouth patches of the form n x 1 x h x w corresponding mouth part
      # After this index_x/y will be n x 1 x h x w
      index_y = index[:,:,0] + torch.from_numpy(np.arange(-40,40)).view(1,1,80).to(device)
      index_y = index_y.unsqueeze(-1).repeat_interleave(512, dim=-1)

      index_x = index[:,:,1] + torch.from_numpy(np.arange(-40,40)).view(1,1,80).to(device)
      index_x = index_x.unsqueeze(-2).repeat_interleave(80, dim=-2)

      # Get patch images (n x 1 x c x h x w)
      patch_images = torch.gather(repeated_images, dim=-2, index_y.unsqueeze(2).repeat_interleave(3,dim=2) )
      patch_images = torch.gather(patch_images, dim=-1, index_x.unsqueeze(2).repeat_interleave(3,dim=2) )

      # Get patch labels (n x 1 x l x h x w)
      patch_labels = torch.gather(repeated_labels, dim=-2, index_y.unsqueeze(2).repeat_interleave(l,dim=2) )
      patch_labels = torch.gather(patch_labels, dim=-1, index_x.unsqueeze(2).repeat_interleave(l,dim=2) )

      # Save patches
      save_patches(patch_images, patch_labels, orig_dataset, indexs)


# Globals
root_dir='data/facial_parts'
dic = {0: 'eyebrow1', 1:'eyebrow2', 2:'eye1', 3:'eye2', 4:'nose', 5:'mouth'}

# Clean first
if shutil.os.path.exists(root_dir):
  shutil.rmtree(root_dir)
shutil.os.mkdir(root_dir)

# Create directories
for k in dic:
  shutil.os.mkdir(shutil.os.path.join(root_dir, dic[k]))
  shutil.os.mkdir(shutil.os.path.join(root_dir, dic[k], 'images'))
  shutil.os.mkdir(shutil.os.path.join(root_dir, dic[k], 'labels'))


# Extract and save facial parts in batch
extract_parts(train_loader, unresized_train)
extract_parts(valid_loader, unresized_valid)
extract_parts(test_loader, unresized_test)