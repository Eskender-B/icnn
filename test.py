import torch
import torch.nn as nn
from model import ICNN
from utils import LOG_INFO
from preprocess import Rescale, ToTensor, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import argparse


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
LOG_INFO('Finally, test loss = %.4f' % (test_loss))