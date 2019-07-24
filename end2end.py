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
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
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
                        shuffle=True, num_workers=4)

unresized_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=ToTensor())



#################################################################
######################### Load models ###########################

model = pickle.load(open('res/saved-model.pth', 'rb'))
model = model.to(device)

names = ['eyebrow', 'eye', 'nose', 'mouth']
models={}
for name in names:
  models[name] = pickle.load(open('res/saved-model-%s.pth'%name, 'rb'))
  models[name].to(device)


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
  box_size = 1024
  res = {}
  offsets = []
  shapes = []
  for i,idx in enumerate(indexs):

    l,h,w = orig_dataset[idx]['labels'].shape
    offset_y, offset_x = (box_size-h)//2, (box_size-w)//2
    offsets.append((offset_y, offset_x))
    shapes.append((h,w))

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

  orig = {'images': orig_images, 'labels':orig_labels}

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

  res['non-mouth'] = {'patch_images': patch_images, 'patch_labels': patch_labels}


  
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

  res['mouth'] = {'patch_images': patch_images, 'patch_labels': patch_labels}

  return res, centroids.long(), orig, np.array(offsets), np.array(shapes)


def bg(labels, fg_indexes):
  """Prepares mask labels for the desired facial part"""
  bg_indexes = list( set(range(11)) - set(fg_indexes) )

  res = torch.cat( [labels.index_select(1, torch.tensor(fg_indexes).long().to(labels.device)),
                    labels.index_select(1, torch.tensor(bg_indexes).long().to(labels.device)).sum(1, keepdim=True).clamp(0.,255.)], 1 )

  return F.one_hot(res.argmax(dim=1), len(fg_indexes)+1).transpose(3,1).transpose(2,3)



def prepare_batches(parts):
  batches = {}

  # Non-mouth parts
  patches, labels = parts['non-mouth']['patch_images'], parts['non-mouth']['patch_labels']

  batches['eyebrow'] = {'image': torch.cat( [patches[:,0,:,:,:], patches[:,1,:,:,:].flip(-1) ]),
                        'labels': torch.cat( [bg(labels[:,0,:,:,:], [2]), bg(labels[:,1,:,:,:], [3]).flip(-1) ] ) }

  batches['eye'] = {'image': torch.cat( [patches[:,2,:,:,:], patches[:,3,:,:,:].flip(-1) ]),
                    'labels': torch.cat( [bg(labels[:,2,:,:,:], [4]), bg(labels[:,3,:,:,:], [5]).flip(-1) ] ) }

  batches['nose'] = {'image': patches[:,4,:,:,:],
                     'labels': bg(labels[:,4,:,:,:], [6]) }


  # Mouth parts
  patches, labels = parts['mouth']['patch_images'], parts['mouth']['patch_labels']

  batches['mouth'] = {'image': patches[:,0,:,:,:],
                   'labels': bg(labels[:,0,:,:,:], [7,8,9]) }

  return batches


def combine_results(pred_labels, orig, centroids):

  colors = torch.tensor([[255,0,0], [255,0,0], [0,0,255], [0,0,255], [255,165,0], [0,255,255], [0,255,0], [255,0,255]]).to(device)
  orig_images, orig_labels = orig['images'].transpose(1,3).transpose(1,2), F.one_hot(orig['labels'].argmax(dim=1), 11).transpose(3,1).transpose(2,3)
  orig_mask = orig_labels.index_select(1, torch.tensor(range(2,10)).to(device)).unsqueeze(-1) * colors.view(1,8,1,1,3)
  orig_mask = orig_mask.sum(1) # May need to fix here


  # parts
  batch_size = args.batch_size
  eyebrow1, eyebrow2 = pred_labels['eyebrow'][0:batch_size,0,:,:], pred_labels['eyebrow'][batch_size:,0,:,:].flip(-1)
  eye1, eye2 = pred_labels['eye'][0:batch_size,0,:,:], pred_labels['eye'][batch_size:,0,:,:].flip(-1)
  nose = pred_labels['nose'][:,0,:,:]
  upper_lip = pred_labels['mouth'][:,0,:,:]
  inner_mouth = pred_labels['mouth'][:,1,:,:]
  lower_lip = pred_labels['mouth'][:,2,:,:]

  centroids = centroids.to('cpu').numpy()
  pred_mask = torch.zeros_like(orig_images, dtype=torch.long)

  for i in range(batch_size):
    # Non-mouth parts
    y, x = centroids[i][0]
    pred_mask[i,y-32:y+32,x-32:x+32,:] += eyebrow1[i].unsqueeze(-1) * colors[0].view(1,1,3)

    y, x = centroids[i][1]
    pred_mask[i,y-32:y+32,x-32:x+32,:] += eyebrow2[i].unsqueeze(-1) * colors[1].view(1,1,3)

    y, x = centroids[i][2]
    pred_mask[i,y-32:y+32,x-32:x+32,:] += eye1[i].unsqueeze(-1) * colors[2].view(1,1,3)

    y, x = centroids[i][3]
    pred_mask[i,y-32:y+32,x-32:x+32,:] += eye2[i].unsqueeze(-1) * colors[3].view(1,1,3)

    y, x = centroids[i][4]
    pred_mask[i,y-32:y+32,x-32:x+32,:] += nose[i].unsqueeze(-1) * colors[4].view(1,1,3)


    # Mouth parts
    y, x = centroids[i][5]
    pred_mask[i,y-40:y+40,x-40:x+40,:] += upper_lip[i].unsqueeze(-1) * colors[5].view(1,1,3)

    y, x = centroids[i][6]
    pred_mask[i,y-40:y+40,x-40:x+40,:] += inner_mouth[i].unsqueeze(-1) * colors[6].view(1,1,3)

    y, x = centroids[i][7]
    pred_mask[i,y-40:y+40,x-40:x+40,:] += lower_lip[i].unsqueeze(-1) * colors[7].view(1,1,3)

  alpha = 0.1
  pred_mask = pred_mask.float()
  orig_mask = orig_mask.float()
  ground_result = torch.where(orig_mask==torch.tensor([0., 0., 0.]).to(device), orig_images, alpha*orig_images + (1.-alpha)*orig_mask)
  pred_result = torch.where(pred_mask==torch.tensor([0., 0., 0.]).to(device), orig_images, alpha*orig_images + (1.-alpha)*pred_mask)  

  return ground_result, pred_result

def save_results(ground, pred, indexs, offsets, shapes):

  ground = np.uint8(ground.clamp(0., 255.).to('cpu').numpy())
  pred = np.uint8(pred.detach().clamp(0.,255.).to('cpu').numpy())

  for i,idx in enumerate(indexs):
    y1,x1 = offsets[i]
    y2,x2 = offsets[i] + shapes[i]
    plt.figure(figsize=(12.8, 9.6))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Ground Truth")
    plt.imshow(ground[i,y1:y2,x1:x2,:])

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Predicted")
    plt.imshow(pred[i,y1:y2,x1:x2,:])

    plt.savefig('res/'+unresized_dataset.name_list[idx, 1].strip() + '.jpg')
    plt.close()


TP = {'eyebrow':0, 'eye':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0}
FP = {'eyebrow':0, 'eye':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0}
TN = {'eyebrow':0, 'eye':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0}
FN = {'eyebrow':0, 'eye':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0}

def calculate_F1(batches, pred_labels):
  global TP, FP, TN, FN
  for name in ['eyebrow', 'eye', 'nose']:
    TP[name]+= (batches[name]['labels'][:,0,:,:] * pred_labels[name][:,0,:,:]).sum().tolist()
    FP[name]+= (batches[name]['labels'][:,1,:,:] * pred_labels[name][:,0,:,:]).sum().tolist()
    TN[name]+= (batches[name]['labels'][:,1,:,:] * pred_labels[name][:,1,:,:]).sum().tolist()
    FN[name]+= (batches[name]['labels'][:,0,:,:] * pred_labels[name][:,1,:,:]).sum().tolist()

  ground = torch.cat( [batches['mouth']['labels'].index_select(1, torch.tensor([0]).to(device)), batches['mouth']['labels'].index_select(1, torch.tensor([1,2,3]).to(device)).sum(1, keepdim=True)], 1)
  pred = torch.cat( [pred_labels['mouth'].index_select(1, torch.tensor([0]).to(device)), pred_labels['mouth'].index_select(1, torch.tensor([1,2,3]).to(device)).sum(1, keepdim=True)], 1)
  TP['u_lip']+= (ground[:,0,:,:] * pred[:,0,:,:]).sum().tolist()
  FP['u_lip']+= (ground[:,1,:,:] * pred[:,0,:,:]).sum().tolist()
  TN['u_lip']+= (ground[:,1,:,:] * pred[:,1,:,:]).sum().tolist()
  FN['u_lip']+= (ground[:,0,:,:] * pred[:,1,:,:]).sum().tolist()

  ground = torch.cat( [batches['mouth']['labels'].index_select(1, torch.tensor([1]).to(device)), batches['mouth']['labels'].index_select(1, torch.tensor([0,2,3]).to(device)).sum(1, keepdim=True)], 1)
  pred = torch.cat( [pred_labels['mouth'].index_select(1, torch.tensor([1]).to(device)), pred_labels['mouth'].index_select(1, torch.tensor([0,2,3]).to(device)).sum(1, keepdim=True)], 1)
  TP['i_mouth']+= (ground[:,0,:,:] * pred[:,0,:,:]).sum().tolist()
  FP['i_mouth']+= (ground[:,1,:,:] * pred[:,0,:,:]).sum().tolist()
  TN['i_mouth']+= (ground[:,1,:,:] * pred[:,1,:,:]).sum().tolist()
  FN['i_mouth']+= (ground[:,0,:,:] * pred[:,1,:,:]).sum().tolist()

  ground = torch.cat( [batches['mouth']['labels'].index_select(1, torch.tensor([2]).to(device)), batches['mouth']['labels'].index_select(1, torch.tensor([0,1,3]).to(device)).sum(1, keepdim=True)], 1)
  pred = torch.cat( [pred_labels['mouth'].index_select(1, torch.tensor([2]).to(device)), pred_labels['mouth'].index_select(1, torch.tensor([0,1,3]).to(device)).sum(1, keepdim=True)], 1)
  TP['l_lip']+= (ground[:,0,:,:] * pred[:,0,:,:]).sum().tolist()
  FP['l_lip']+= (ground[:,1,:,:] * pred[:,0,:,:]).sum().tolist()
  TN['l_lip']+= (ground[:,1,:,:] * pred[:,1,:,:]).sum().tolist()
  FN['l_lip']+= (ground[:,0,:,:] * pred[:,1,:,:]).sum().tolist()


def show_F1():
  F1 = {}
  PRECISION = {}
  RECALL = {}
  for key in TP:
    PRECISION[key] = float(TP[key]) / (TP[key] + FP[key])
    RECALL[key] = float(TP[key]) / (TP[key] + FN[key])
    F1[key] = 2.*PRECISION[key]*RECALL[key]/(PRECISION[key]+RECALL[key])

  print("\n\n", "PART ", "F1-MEASURE ", "PRECISION ", "RECALL")
  for k in F1:
    print("%s\t"%k, "%.4f\t"%F1[k], "%.4f\t"%PRECISION[k], "%.4f\t"%RECALL[k])


with torch.no_grad():
  for batch in test_loader:
    images, labels, indexs = batch['image'].to(device), batch['labels'].to(device), batch['index']

    ## Calculate locations of facial parts
    pred_labels = F.softmax(model(images), 1)
    centroids = calculate_centroids(pred_labels)

    ## Extract patches from face from their location given in centroids
    # Get also shift-scaled centroids, offsets and shapes 
    parts, centroids, orig, offsets, shapes = extract_parts(indexs, centroids, unresized_dataset)

    ## Prepare batches for facial parts
    batches = prepare_batches(parts)
   
    ## Get prediction
    pred_labels = {}
    for name in batches:
      pred_labels[name] = F.one_hot(models[name](batches[name]['image']).argmax(dim=1), models[name].L).transpose(3,1).transpose(2,3)

    ## Update F1-measure stat for this batch
    calculate_F1(batches, pred_labels)

    ## Rearrange patch results onto original image
    ground_result, pred_result = combine_results(pred_labels, orig, centroids)

    ## Save results
    save_results(ground_result, pred_result, indexs, offsets, shapes)
    print("Processed %d images"%args.batch_size)

## Show stats
show_F1()