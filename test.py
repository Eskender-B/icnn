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

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=25, type=int, help="Batch size to use during training.")
args = parser.parse_args()
print(args)


if torch.cuda.is_available():
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=transforms.Compose([
                                               Rescale((64,64)),
                                               ToTensor()
                                           ]))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=1)

unresized_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
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

model = ICNN()
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

model.load_state_dict(torch.load('saved-model.pth'))
test_loss = evaluate(model, test_loader, criterion)
LOG_INFO('test loss = %.4f' % (test_loss))


###### See some images ######
#colors = torch.Tensor([(255,255,255),	(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255),(192,192,192)]).to(device)

def show_centroids(image, centroids1, centroids2):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(centroids[:, 0], centroids1[:, 1], s=10, marker='x', c='r')
    plt.scatter(centroids[:, 0], centroids2[:, 1], s=10, marker='x', c='g')
    plt.pause(0.001)  # pause a bit so that plots are updated

def calculate_centroids(tensor):
	tensor = tensor.float()
	h = tensor.shape[2]
	w = tensor.shape[3]
	indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
	indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
	
	center_y = tensor.mean(3) * indexs_y.view(1,1,-1)
	center_y = center_y.mean(2, keepdim=True)
	
	center_x = tensor.mean(2) * indexs_y.view(1,1,-1)
	center_x = center_x.mean(2, keepdim=True)

	return torch.cat([center_x, center_y], 2)

n_test = 4
indxs = np.random.randint(len(test_dataset), size=n_test)

unresized_images = [unresized_dataset[i]['image'] for i in indxs]

images = torch.stack([test_dataset[i]['image'] for i in indxs]).to(device)
orig_labels = np.array([test_dataset[i]['labels'].numpy() for i in indxs])
pred_labels = torch.softmax(model(images), 1)

orig_centroids = calculate_centroids(torch.softmax(torch.from_numpy(orig_labels),1))	# get centroid and scale back
orig_centroids = np.array( orig_centroids.detach().to('cpu').numpy(), np.int)

centroids = 4 *calculate_centroids(pred_labels)	# get centroid and scale back
centroids = np.array( centroids.detach().to('cpu').numpy(), np.int)


for i in range(n_test):
	ax = plt.subplot(2, 2, i + 1)
	plt.tight_layout()
	ax.set_title('Sample #{}'.format(i))
	ax.axis('off')
	show_centroids(unresized_images[i], centroids[i], orig_centroids[i])

plt.savefig('test_outputs.jpg')