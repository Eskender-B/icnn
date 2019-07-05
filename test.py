import torch
import torch.nn as nn
from model import ICNN
from utils import LOG_INFO
from preprocess import Rescale, ToTensor, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import argparse
import numpy as np
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
                        shuffle=True, num_workers=4)

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
n_test = 5
indxs = np.random.randint(len(test_dataset), size=n_test)

images = torch.stack([test_dataset[i]['image'] for i in indxs]).to(device)
orig_labels = np.array([test_dataset[i]['labels'].numpy() for i in indxs]).argmax(axis=1)

pred_labels = model(images).argmax(dim=1, keepdim=False).to('cpu')
images = images.transpose(1,2).transpose(2,3).to('cpu')


for i in range(n_test):
	ax = plt.subplot(3, n_test, i + 1)
	plt.tight_layout()
	ax.set_title('Sample #{}'.format(i))
	ax.axis('off')
	plt.imshow(images[i])

	ax = plt.subplot(3, n_test, n_test+i + 1)
	plt.tight_layout()
	ax.set_title('Sample #{}'.format(i))
	ax.axis('off')
	plt.imshow(pred_labels[i])

	ax = plt.subplot(3, n_test, 2*n_test+i + 1)
	plt.tight_layout()
	ax.set_title('Sample #{}'.format(i))
	ax.axis('off')
	plt.imshow(orig_labels[i])

plt.savefig('test_outputs.jpg')

