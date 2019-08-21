import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from preprocess import Rescale, ToTensor, ImageDataset, Invert, DataArg
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
def make_dataset(file, dir_name, trans, fg_indexs=None, bg_indexs=set([])):
	return ImageDataset(txt_file=file, root_dir='data/facial_parts/'+dir_name, fg_indexs=fg_indexs, bg_indexs=bg_indexs,
                                           transform=transforms.Compose(trans))                                           	

train_datasets = {}
valid_datasets = {}
test_datasets = {}

## Training set
train_datasets['eyebrow'] = ConcatDataset([make_dataset('exemplars.txt', 'eyebrow1', fg_indexs=set([2]), trans=[DataArg(), ToTensor()]), 
										   make_dataset('exemplars.txt', 'eyebrow2', fg_indexs=set([3]), trans=[Invert(), DataArg(), ToTensor()])])

train_datasets['eye'] = ConcatDataset([make_dataset('exemplars.txt', 'eye1', fg_indexs=set([4]), trans=[DataArg(),ToTensor()]), 
									make_dataset('exemplars.txt', 'eye2', fg_indexs=set([5]), trans=[Invert(),DataArg(), ToTensor()])])

train_datasets['nose'] = make_dataset('exemplars.txt', 'nose', fg_indexs=set([6]), trans=[DataArg(),ToTensor()])
train_datasets['mouth'] = make_dataset('exemplars.txt', 'mouth', fg_indexs=set([7,8,9]), trans=[DataArg(), ToTensor()])


## Validation set
valid_datasets['eyebrow'] = ConcatDataset([make_dataset('tuning.txt', 'eyebrow1', fg_indexs=set([2]), trans=[ToTensor()]), 
										   make_dataset('tuning.txt', 'eyebrow2', fg_indexs=set([3]), trans=[ToTensor(), Invert()])])

valid_datasets['eye'] = ConcatDataset([make_dataset('tuning.txt', 'eye1', fg_indexs=set([4]), trans=[ToTensor()]), 
									make_dataset('tuning.txt', 'eye2', fg_indexs=set([5]), trans=[ToTensor(), Invert()])])

valid_datasets['nose'] = make_dataset('tuning.txt', 'nose', fg_indexs=set([6]), trans=[ToTensor()])
valid_datasets['mouth'] = make_dataset('tuning.txt', 'mouth', fg_indexs=set([7,8,9]), trans=[ToTensor()])


## Testing set
test_datasets['eyebrow'] = ConcatDataset([make_dataset('testing.txt', 'eyebrow1', fg_indexs=set([2]), trans=[ToTensor()]), 
										   make_dataset('testing.txt', 'eyebrow2', fg_indexs=set([3]), trans=[ToTensor(), Invert()])])

test_datasets['eye'] = ConcatDataset([make_dataset('testing.txt', 'eye1', fg_indexs=set([4]), trans=[ToTensor()]), 
									make_dataset('testing.txt', 'eye2', fg_indexs=set([5]), trans=[ToTensor(), Invert()])])

test_datasets['nose'] = make_dataset('testing.txt', 'nose', fg_indexs=set([6]), trans=[ToTensor()])
test_datasets['mouth'] = make_dataset('testing.txt', 'mouth', fg_indexs=set([7,8,9]), trans=[ToTensor()])




####################################
############ ICNN Models ############
models = {}
optimizers={}
schedulers={}
models['eyebrow'] = ICNN(output_maps=2).to(device)
models['eye'] = ICNN(output_maps=2).to(device)
models['nose'] = ICNN(output_maps=2).to(device)
models['mouth'] = ICNN(output_maps=4).to(device)

optimizers['eyebrow'] = optim.Adam(models['eyebrow'].parameters(), lr=args.lr)
schedulers['eyebrow'] = optim.lr_scheduler.StepLR(optimizers['eyebrow'], step_size=50, gamma=0.5)

optimizers['eye'] = optim.Adam(models['eye'].parameters(), lr=args.lr)
schedulers['eye'] = optim.lr_scheduler.StepLR(optimizers['eye'], step_size=50, gamma=0.5)

optimizers['nose'] = optim.Adam(models['nose'].parameters(), lr=args.lr)
schedulers['nose'] = optim.lr_scheduler.StepLR(optimizers['nose'], step_size=50, gamma=0.5)

optimizers['mouth'] = optim.Adam(models['mouth'].parameters(), lr=args.lr)
schedulers['mouth'] = optim.lr_scheduler.StepLR(optimizers['mouth'], step_size=50, gamma=0.5)


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



def train_model(part_name, criterion, epochs):
	LOSS = 100
	epoch_min = 1

	LOG_INFO('\n\nTraining model for %s ...' % part_name)
	model = models[part_name]
	optimizer = optimizers[part_name]
	scheduler = schedulers[part_name]

	train_loader = DataLoader(train_datasets[part_name], batch_size=args.batch_size, shuffle=True, num_workers=4)
	valid_loader = DataLoader(valid_datasets[part_name], batch_size=args.batch_size, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_datasets[part_name], batch_size=args.batch_size, shuffle=True, num_workers=4)

	for epoch in range(1, epochs + 1):
		train(epoch, model, train_loader, optimizer, criterion)
		valid_loss = evaluate(model, valid_loader, criterion)
		if valid_loss < LOSS:
			LOSS = valid_loss
			epoch_min = epoch
			pickle.dump(model, open('res/saved-model-%s.pth'%part_name, 'wb'))

		#scheduler.step()
		msg = '...Epoch %02d, val loss (%s) = %.4f' % (epoch, 	part_name, valid_loss)
		LOG_INFO(msg)


	model = pickle.load(open('res/saved-model-%s.pth'%part_name, 'rb'))
	msg = 'Min @ Epoch %02d, val loss (%s) = %.4f' % (epoch_min, 	part_name, LOSS)
	LOG_INFO(msg)

	test_loss = evaluate(model, test_loader, criterion)
	LOG_INFO('Finally, test loss (%s) = %.4f' % (part_name, test_loss))



names = ['eyebrow', 'eye', 'nose', 'mouth']
#names = ['mouth']
for name in names:
	if name=='mouth':
		train_model(name, criterion, 2*args.epochs)
	else:
		train_model(name, criterion, args.epochs)