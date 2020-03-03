import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from preprocess import Rescale, ToTensor, ImageDataset, DataArg, FaceDetect
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from model import ICNN
import argparse
from utils import LOG_INFO
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train")
parser.add_argument("--load_model", default=False, type=bool, help="Load saved-model")
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

resize_num = 64
# Load data
train_dataset = ImageDataset(txt_file='exemplars.txt',
                                            root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0,1,10]),
                                           transform=transforms.Compose([
                                               Rescale((resize_num,resize_num)),
                                               ToTensor()
                                           ]))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


valid_dataset = ImageDataset(txt_file='tuning.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0,1,10]),
                                           transform=transforms.Compose([
                                              Rescale((resize_num,resize_num)),
                                               ToTensor()
                                           ]))
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0,1,10]),
                                           transform=transforms.Compose([
                                               Rescale((resize_num,resize_num)),
                                               ToTensor(),
                                           ]))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)




####################################
############ ICNN Model ############

model = ICNN(output_maps=9)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.CrossEntropyLoss()
model = model.to(device)
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

LOSS = 100
epoch_min = 1

if args.load_model == True:
	model = pickle.load(open('res/saved-model.pth', 'rb'))


for epoch in range(1, args.epochs + 1):
	train(epoch, model, train_loader, optimizer, criterion)
	valid_loss = evaluate(model, valid_loader, criterion)
	if valid_loss < LOSS:
		LOSS = valid_loss
		epoch_min = epoch
		pickle.dump(model, open('res/saved-model.pth', 'wb'))

	#scheduler.step()
	msg = '...Epoch %02d, val loss = %.4f' % (epoch, valid_loss)
	LOG_INFO(msg)


model = pickle.load(open('res/saved-model.pth', 'rb'))
msg = 'Min @ Epoch %02d, val loss = %.4f' % (epoch_min, LOSS)
LOG_INFO(msg)
test_loss = evaluate(model, test_loader, criterion)
LOG_INFO('Finally, test loss = %.4f' % (test_loss))