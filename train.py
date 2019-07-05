import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from preprocess import Rescale, ToTensor, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from model import ICNN
import argparse
from utils import LOG_INFO


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=25, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=1, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=10, type=float, help="Learning rate for optimizer")
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

# Load data
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
valid_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           transform=transforms.Compose([
                                               Rescale((64,64)),
                                               ToTensor()
                                           ]))
test_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)




####################################
############ ICNN Model ############

model = ICNN()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)
#####################################



def train(epoch, model, train_loader, optimizer, criterion):
	loss_list = []
	model.train()

	for i, batch in enumerate(train_loader):
		optimizer.zero_grad()
		image, labels = batch['image'].to(device), batch['labels'].to(device)
		predictions = model(image)
		loss = criterion(predictions, labels.argmax(dim=1, keepdim=False))

		loss.backward()
		optimizer.step()


		loss_list.append(loss.item())


		if i % args.display_freq == 0:
			msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f" % (
			epoch, i, len(train_loader), np.mean(loss_list))
			LOG_INFO(msg)
			loss_list.clear()


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



for epoch in range(1, args.epochs + 1):
	train(epoch, model, train_loader, optimizer, criterion)
	valid_loss = evaluate(model, valid_loader, criterion)
	msg = '...Epoch %02d, val loss = %.4f' % (
	epoch, valid_loss)
	LOG_INFO(msg)

torch.save(model.state_dict(), 'saved-model.pth')

LOG_INFO('Test best model @ Epoch %02d' % best_epoch)
model.load_state_dict(torch.load('saved-model.pth'))
test_loss = evaluate(model, test_loader, criterion)
LOG_INFO('Finally, test loss = %.4f' % (test_loss))