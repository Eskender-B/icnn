import torch
import torch.nn as nn
from model import ICNN
from bg_modulate import Modulator
from utils import LOG_INFO
from preprocess import Rescale, ToTensor, ImageDataset, FaceDetect
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import argparse
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
import shutil
import pickle
from skimage import transform

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

resize_num = 64
part_width = 64
part_mouth = 80
## Load test data 
test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0,1,10]),
                                           transform=transforms.Compose([
                                               Rescale((resize_num,resize_num)),
                                               ToTensor(),
                                           ]))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

unresized_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=ToTensor(), calc_bg=False)


## Load models
model = pickle.load(open('res/saved-model.pth', 'rb'))
model = model.to(device)

names = ['eyebrow', 'eye', 'nose', 'mouth']
models={}
modulators={}
for name in names:
  models[name] = pickle.load(open('res/saved-model-%s.pth'%name, 'rb'))
  modulators[name] = pickle.load(open('res/saved-modulator-%s.pth'%name, 'rb'))
  #models[name].to(device)


#################################################################
############### Helper functions ################################

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



def map_func1(landmarks, coords):
  dst = np.array([[-0.25,-0.1], [0.25, -0.1], [0.0, 0.1], [-0.15, 0.4], [0.15, 0.4]])
  tform = transform.estimate_transform('similarity', np.array(landmarks, np.float), dst)
  tform2 = transform.SimilarityTransform(scale=1/32, rotation=0, translation=(-1.0, -1.0))
  ans = tform.inverse(tform2(coords))
  #print(coords, ans)
  return ans

def extract_parts(indexs, centroids, orig_dataset, landmarks=None):
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
    c_box_size = 128
    new_h, new_w = [int(resize_num * h / w), resize_num] if h>w else [resize_num, int(resize_num * w / h)]
    c_offset_y, c_offset_x = (c_box_size-new_h)//2, (c_box_size-new_w)//2

    #centroids[i] =  ( (centroids[i] - torch.Tensor([c_offset_y, c_offset_x]).view(1,2).to(device) )\
    #                             * torch.Tensor([h/new_h, w/new_w]).view(1,2).to(device) ) \
    #                             + torch.Tensor([offset_y, offset_x]).view(1,2).to(device)
    
    if landmarks is not None:
      centroids[i] = torch.tensor(map_func1(landmarks[i], centroids[i].to('cpu').flip(-1) )).to(device).flip(-1) + torch.Tensor([offset_y, offset_x]).view(1,2).to(device)
    else:
      centroids[i] = centroids[i] * torch.Tensor([h/float(resize_num), w/float(resize_num)]).view(1,2).to(device) + torch.Tensor([offset_y, offset_x]).view(1,2).to(device)
      

  orig_images = orig_images.to(device).view(len(indexs),3,box_size,box_size)
  orig_labels = F.one_hot(orig_labels.to(device).view(len(indexs),l,box_size,box_size).argmax(1), 11).transpose(3,1).transpose(2,3)

  orig = {'images': orig_images, 'labels':orig_labels}

  #################
  # Non-Mouth parts
  index = centroids.index_select(1, torch.tensor(range(5)).to(device)).round().long()
  n_parts = index.shape[-2]

  # Construct repeated image of n x p x c x h x w
  repeated_images = orig_images.unsqueeze(1).repeat_interleave(n_parts, dim=1)
  repeated_labels = orig_labels.unsqueeze(1).repeat_interleave(n_parts, dim=1)

  # Calculate index of patches of the form n x p x resize_num x resize_num corresponding to each facial part
  # After this index_x/y will be n x p x resize_num x resize_num
  index_y = index[:,:,0].unsqueeze(-1) + torch.from_numpy(np.arange(-part_width//2,part_width//2)).view(1,1,part_width).to(device)
  index_y = index_y.unsqueeze(-1).repeat_interleave(box_size, dim=-1)

  index_x = index[:,:,1].unsqueeze(-1) + torch.from_numpy(np.arange(-part_width//2,part_width//2)).view(1,1,part_width).to(device)
  index_x = index_x.unsqueeze(-2).repeat_interleave(part_width, dim=-2)

  # Get patch images (n x p x c x h x w)
  patch_images = torch.gather(repeated_images, -2, index_y.unsqueeze(2).repeat_interleave(3,dim=2) )
  patch_images = torch.gather(patch_images, -1, index_x.unsqueeze(2).repeat_interleave(3,dim=2) )

  # Get patch labels (n x p x l x h x w)
  patch_labels = torch.gather(repeated_labels, -2, index_y.unsqueeze(2).repeat_interleave(l,dim=2) )
  patch_labels = torch.gather(patch_labels, -1, index_x.unsqueeze(2).repeat_interleave(l,dim=2) )

  res['non-mouth'] = {'patch_images': patch_images, 'patch_labels': patch_labels}


  
  ##################
  # Mouth part
  index = centroids.index_select(1, torch.tensor(range(5,8)).to(device)).mean(dim=1, keepdim=True).round().long()

  # Construct repeated image of n x 1 x c x h x w
  repeated_images = orig_images.unsqueeze(1)
  repeated_labels = orig_labels.unsqueeze(1)
  
  # Calculate index of mouth patches of the form n x 1 x 80 x 80 corresponding mouth part
  # After this index_x/y will be n x 1 x 80 x 80
  index_y = index[:,:,0].unsqueeze(-1) + torch.from_numpy(np.arange(-part_mouth//2,part_mouth//2)).view(1,1,part_mouth).to(device)
  index_y = index_y.unsqueeze(-1).repeat_interleave(box_size, dim=-1)

  index_x = index[:,:,1].unsqueeze(-1) + torch.from_numpy(np.arange(-part_mouth//2,part_mouth//2)).view(1,1,part_mouth).to(device)
  index_x = index_x.unsqueeze(-2).repeat_interleave(part_mouth, dim=-2)

  # Get patch images (n x 1 x c x 80 x 80)
  patch_images = torch.gather(repeated_images, -2, index_y.unsqueeze(2).repeat_interleave(3,dim=2) )
  patch_images = torch.gather(patch_images, -1, index_x.unsqueeze(2).repeat_interleave(3,dim=2) )

  # Get patch labels (n x 1 x l x 80 x 80)
  patch_labels = torch.gather(repeated_labels, -2, index_y.unsqueeze(2).repeat_interleave(l,dim=2) )
  patch_labels = torch.gather(patch_labels, -1, index_x.unsqueeze(2).repeat_interleave(l,dim=2) )

  res['mouth'] = {'patch_images': patch_images, 'patch_labels': patch_labels}

  return res, centroids, orig, np.array(offsets), np.array(shapes)


def bg(labels, fg_indexes):
  """Prepares mask labels for the desired facial part"""

  fg = labels.index_select(1, torch.tensor(fg_indexes).long().to(labels.device))
  bg = torch.tensor(1).to(labels.device) - fg.sum(1, keepdim=True)
  res = torch.cat( [fg, bg], 1 )

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
  orig_images, orig_labels = orig['images'].transpose(1,3).transpose(1,2) * 255, F.one_hot(orig['labels'].argmax(dim=1), 11).transpose(3,1).transpose(2,3)
  orig_mask = orig_labels.index_select(1, torch.tensor(range(2,10)).to(device)).unsqueeze(-1) * colors.view(1,8,1,1,3)
  orig_mask = orig_mask.sum(1) # May need to fix here


  # parts
  batch_size = pred_labels['nose'].shape[0]
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
    y, x = np.array(centroids[i][0].round(), np.long)
    pred_mask[i,y-part_width//2:y+part_width//2,x-part_width//2:x+part_width//2,:] += eyebrow1[i].unsqueeze(-1) * colors[0].view(1,1,3)

    y, x = np.array(centroids[i][1].round(), np.long)
    pred_mask[i,y-part_width//2:y+part_width//2,x-part_width//2:x+part_width//2,:] += eyebrow2[i].unsqueeze(-1) * colors[1].view(1,1,3)

    y, x = np.array(centroids[i][2].round(), np.long)
    pred_mask[i,y-part_width//2:y+part_width//2,x-part_width//2:x+part_width//2,:] += eye1[i].unsqueeze(-1) * colors[2].view(1,1,3)

    y, x = np.array(centroids[i][3].round(), np.long)
    pred_mask[i,y-part_width//2:y+part_width//2,x-part_width//2:x+part_width//2,:] += eye2[i].unsqueeze(-1) * colors[3].view(1,1,3)

    y, x = np.array(centroids[i][4].round(), np.long)
    pred_mask[i,y-part_width//2:y+part_width//2,x-part_width//2:x+part_width//2,:] += nose[i].unsqueeze(-1) * colors[4].view(1,1,3)


    # Mouth parts
    y, x = np.array(np.array(centroids[i][5:8], dtype=np.float).mean(0).round(), np.long)
    pred_mask[i,y-part_mouth//2:y+part_mouth//2,x-part_mouth//2:x+part_mouth//2,:] += upper_lip[i].unsqueeze(-1) * colors[5].view(1,1,3)

    pred_mask[i,y-part_mouth//2:y+part_mouth//2,x-part_mouth//2:x+part_mouth//2,:] += inner_mouth[i].unsqueeze(-1) * colors[6].view(1,1,3)

    pred_mask[i,y-part_mouth//2:y+part_mouth//2,x-part_mouth//2:x+part_mouth//2,:] += lower_lip[i].unsqueeze(-1) * colors[7].view(1,1,3)

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
  tot_p = 0.0
  tot_r = 0.0
  for key in TP:
    PRECISION[key] = float(TP[key]) / (TP[key] + FP[key] + 0.000001)
    RECALL[key] = float(TP[key]) / (TP[key] + FN[key] + 0.000001)
    F1[key] = 2.*PRECISION[key]*RECALL[key]/(PRECISION[key]+RECALL[key]+0.0000001)

    tot_p += PRECISION[key]
    tot_r += RECALL[key]

  #avg_p = tot_p/len(TP)
  #avg_r = tot_r/len(TP)
  #overall_F1 = 2.* avg_p*avg_r/ (avg_p+avg_r)

  mouth_p = (PRECISION['u_lip'] + PRECISION['i_mouth'] + PRECISION['l_lip'])/3.0
  mouth_r = (RECALL['u_lip'] + RECALL['i_mouth'] + RECALL['l_lip'])/3.0
  mouth_F1 = 2.* mouth_p * mouth_r / (mouth_p+mouth_r+0.0000001)

  avg_p = (PRECISION['eyebrow']+PRECISION['eye']+PRECISION['nose']+mouth_p)/4.0
  avg_r = (RECALL['eyebrow']+RECALL['eye']+RECALL['nose']+mouth_r)/4.0
  overall_F1 = 2.* avg_p*avg_r/ (avg_p+avg_r+0.0000001)


  print("\n\n", "PART\t\t", "F1-MEASURE ", "PRECISION ", "RECALL")
  for k in F1:
    print("%s\t\t"%k, "%.4f\t"%F1[k], "%.4f\t"%PRECISION[k], "%.4f\t"%RECALL[k])

  print("mouth(all)\t", "%.4f\t"%mouth_F1, "%.4f\t"%mouth_p, "%.4f\t"%mouth_r)
  print("Overall\t\t", "%.4f\t"%overall_F1, "%.4f\t"%avg_p, "%.4f\t"%avg_r)



#############################################
################## Main #####################
def main():

  with torch.no_grad():
    for batch in test_loader:
      images, labels, indexs, landmarks = batch['image'].to(device), batch['labels'].to(device), batch['index'], batch['landmarks']

      ## Calculate locations of facial parts from stage1
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
        #pred_labels[name] = F.one_hot(models[name](batches[name]['image']).argmax(dim=1), models[name].L).transpose(3,1).transpose(2,3)
        pred_labels[name] = F.one_hot(modulators[name](models[name](batches[name]['image'])).argmax(dim=1), models[name].L).transpose(3,1).transpose(2,3)
        

      ## Update F1-measure stat for this batch
      calculate_F1(batches, pred_labels)

      ## Rearrange patch results onto original image
      ground_result, pred_result = combine_results(pred_labels, orig, centroids)

      ## Save results
      save_results(ground_result, pred_result, indexs, offsets, shapes)
      print("Processed %d images"%args.batch_size)

  ## Show stats
  show_F1()

if __name__ == '__main__':
  main()