import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from preprocess import Rescale, ToTensor, ImageDataset, Invert
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, utils
from model import ICNN
import argparse
from utils import LOG_INFO
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs to train")
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")


# Load data
def make_dataset(file, dir_name, bg_indexs=set([]), invert=False):
	return ImageDataset(txt_file=file, root_dir='data/facial_parts/'+dir_name, bg_indexs=bg_indexs,
                                           transform=transforms.Compose([ToTensor(), Invert()]) if invert else ToTensor())

train_datasets = {}
valid_datasets = {}
test_datasets = {}

## Training set
train_datasets['eyebrow'] = ConcatDataset([make_dataset('exemplars.txt', 'eyebrow1', set(range(11)).difference([2])), 
										   make_dataset('exemplars.txt', 'eyebrow2', set(range(11)).difference([3]), invert=True)])

train_datasets['eye'] = ConcatDataset([make_dataset('exemplars.txt', 'eye1', set(range(11)).difference([4])), 
									make_dataset('exemplars.txt', 'eye2', set(range(11)).difference([5]), invert=True)])

train_datasets['nose'] = make_dataset('exemplars.txt', 'nose', set(range(11)).difference([6]))
train_datasets['mouth'] = make_dataset('exemplars.txt', 'mouth', set(range(11)).difference([7,8,9]))


## Validation set
valid_datasets['eyebrow'] = ConcatDataset([make_dataset('tuning.txt', 'eyebrow1', set(range(11)).difference([2])), 
										   make_dataset('tuning.txt', 'eyebrow2', set(range(11)).difference([3]), invert=True)])

valid_datasets['eye'] = ConcatDataset([make_dataset('tuning.txt', 'eye1', set(range(11)).difference([4])), 
									make_dataset('tuning.txt', 'eye2', set(range(11)).difference([5]), invert=True)])

valid_datasets['nose'] = make_dataset('tuning.txt', 'nose', set(range(11)).difference([6]))
valid_datasets['mouth'] = make_dataset('tuning.txt', 'mouth', set(range(11)).difference([7,8,9]))


## Testing set
test_datasets['eyebrow'] = ConcatDataset([make_dataset('testing.txt', 'eyebrow1', set(range(11)).difference([2])), 
										   make_dataset('testing.txt', 'eyebrow2', set(range(11)).difference([3]), invert=True)])

test_datasets['eye'] = ConcatDataset([make_dataset('testing.txt', 'eye1', set(range(11)).difference([4])), 
									make_dataset('testing.txt', 'eye2', set(range(11)).difference([5]), invert=True)])

test_datasets['nose'] = make_dataset('testing.txt', 'nose', set(range(11)).difference([6]))
test_datasets['mouth'] = make_dataset('testing.txt', 'mouth', set(range(11)).difference([7,8,9]))




####################################
############ ICNN Models ############
models = {}
optimizers={}
schedulers={}
models['eyebrow'] = ICNN(output_maps=2).to(device)
models['eye'] = ICNN(output_maps=2).to(device)
models['nose'] = ICNN(output_maps=2).to(device)
models['mouth'] = ICNN(output_maps=4).to(device)

optimizers['eyebrow'] = optim.SGD(models['eyebrow'].parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
schedulers['eyebrow'] = optim.lr_scheduler.StepLR(optimizers['eyebrow'], step_size=15, gamma=0.5)

optimizers['eye'] = optim.SGD(models['eye'].parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
schedulers['eye'] = optim.lr_scheduler.StepLR(optimizers['eye'], step_size=15, gamma=0.5)

optimizers['nose'] = optim.SGD(models['nose'].parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
schedulers['nose'] = optim.lr_scheduler.StepLR(optimizers['nose'], step_size=15, gamma=0.5)

optimizers['mouth'] = optim.SGD(models['mouth'].parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
schedulers['mouth'] = optim.lr_scheduler.StepLR(optimizers['mouth'], step_size=15, gamma=0.5)


criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.xavier_normal_(m.bias.data)

#model.apply(weights_init)
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

def train_model(part_name, criterion):

	LOG_INFO('\n\nTraining model for %s ...' % part_name)
	model = models[part_name]
	optimizer = optimizers[part_name]
	scheduler = schedulers[part_name]

	train_loader = DataLoader(train_datasets[part_name], batch_size=args.batch_size, shuffle=True, num_workers=4)
	valid_loader = DataLoader(valid_datasets[part_name], batch_size=args.batch_size, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_datasets[part_name], batch_size=args.batch_size, shuffle=True, num_workers=4)

	for epoch in range(1, args.epochs + 1):
		train(epoch, model, train_loader, optimizer, criterion)
		valid_loss = evaluate(model, valid_loader, criterion)
		scheduler.step()
		msg = '...Epoch %02d, val loss (%s) = %.4f' % (
		epoch, 	part_name, valid_loss)
		LOG_INFO(msg)

	pickle.dump(model, open('res/saved-model-%s.pth'%part_name, 'wb'))
	model = pickle.load(open('res/saved-model-%s.pth'%part_name, 'rb'))

	test_loss = evaluate(model, test_loader, criterion)
	LOG_INFO('Finally, test loss (%s) = %.4f' % (part_name, test_loss))



names = ['eyebrow', 'eye', 'nose', 'mouth']
for name in names:
	train_model(name, criterion)