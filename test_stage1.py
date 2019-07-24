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
                                           bg_indexs=set([0,1,10]),
                                           transform=None)


def evaluate(model, loader, criterion):
	epoch_loss = 0
	model.eval()

	with torch.no_grad():
		for batch in loader:
			image, labels = batch['image'].to(device), batch['labels'].to(device)
			predictions = model(image)
			loss = criterion(predictions, labels.argmax(dim=1, keepdim=False))

			epoch_loss += loss.item()

	return epoch_loss / len(loader)


criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

model = pickle.load(open('res/saved-model.pth', 'rb'))
model = model.to(device)
test_loss = evaluate(model, test_loader, criterion)
LOG_INFO('test loss = %.4f' % (test_loss))


###### See some images ######
#colors = torch.Tensor([(255,255,255),	(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255),(192,192,192)]).to(device)

def show_centroids(image, centroids_pred, centroids_orig):
    """Show image with landmarks"""
    h,w,c = image.shape
    plt.imshow(image)
    plt.scatter(w/64*centroids_orig[:-1, 0], h/64*centroids_orig[:-1, 1], s=10, marker='x', c='r')
    plt.scatter(w/64*centroids_pred[:-1, 0], h/64*centroids_pred[:-1, 1], s=10, marker='x', c='g')
    #plt.pause(0.001)  # pause a bit so that plots are updated

def calculate_centroids(tensor):
	tensor = tensor.float() + 1e-10
	n,l,h,w = tensor.shape
	indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
	indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
	center_y = tensor.sum(3) * indexs_y.view(1,1,-1) 
	center_y = center_y.sum(2, keepdim=True) / tensor.sum([2,3]).view(n,l,1)
	center_x = tensor.sum(2) * indexs_x.view(1,1,-1)
	center_x = center_x.sum(2, keepdim=True) / tensor.sum([2,3]).view(n,l,1)
	return torch.cat([center_x, center_y], 2)


dist_error = np.zeros(9) # For each face part, last is background
count=0
def update_error(pred_centroids, orig_centroids):
	global dist_error, count
	count+= pred_centroids.shape[0]
	dist_error += torch.pow(pred_centroids - orig_centroids, 2).sum(dim=2).sqrt().sum(dim=0).to('cpu').numpy()


def show_error():
	global dist_error
	dist_error /=count
	parts = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
	
	print("\n\nDistance Error in 64 X 64 image (in pixels) ... ")
	for i in range(len(parts)-1):
		print(parts[i], "%.2f pixels"%dist_error[i])
	print(parts[-1], "%.2f pixels"%dist_error[-3:].mean())

	print("Total Error: %.2f"%dist_error.mean())


def save_results(indexs, pred_centroids, orig_centroids):
	pred_centroids = pred_centroids.detach().to('cpu').numpy()
	orig_centroids = orig_centroids.to('cpu').numpy()

	for i,idx in enumerate(indexs):
		img = unresized_dataset[idx]['image']
		h,w,c = img.shape
		plt.imshow(img)

		plt.scatter(w/64*orig_centroids[i,:-1, 0], h/64*orig_centroids[i,:-1, 1], s=10, marker='x', c='r', label='Ground Truth')
		plt.scatter(w/64*pred_centroids[i,:-1, 0], h/64*pred_centroids[i,:-1, 1], s=10, marker='x', c='g', label='Predicted')

		plt.legend()
		plt.savefig('res/'+unresized_dataset.name_list[idx, 1].strip() + '_loc.jpg')
		plt.close()


with torch.no_grad():
	for batch in test_loader:
		images, labels, indexs = batch['image'].to(device), batch['labels'].to(device), batch['index']

		# Original Centroids
		orig_labels = F.normalize(labels, 1)
		orig_centroids = calculate_centroids(orig_labels)

		# Predicted Centroids
		pred_labels = F.softmax(model(images), 1)
		pred_centroids = calculate_centroids(pred_labels)	

		# Update error stat
		update_error(pred_centroids, orig_centroids)

		# Save results
		save_results(indexs, pred_centroids, orig_centroids)

show_error()