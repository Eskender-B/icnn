import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from preprocess import Rescale, ToTensor, ImageDataset, Invert, DataArg
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, utils
from model import IRCNN, ICNN
import argparse
from utils import LOG_INFO
import pickle
import numpy as np


LR = 0.1
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

# Load data
def make_dataset(file, dir_name, trans, fg_indexs=None, bg_indexs=set([])):
	return ImageDataset(txt_file=file, root_dir='data/facial_parts/'+dir_name, fg_indexs=fg_indexs, bg_indexs=bg_indexs,
                                           transform=transforms.Compose(trans))                                           	



criterion = nn.CrossEntropyLoss()

class Modulator(nn.Module):

	def __init__(self, num):
		super(Modulator,self).__init__()
		self.alpha = nn.Parameter(torch.Tensor(num))
		self.beta = nn.Parameter(torch.Tensor(1))
		nn.init.ones_(self.alpha)
		nn.init.zeros_(self.beta)



	def forward(self, inp):
		n,l,h,w = inp.shape
		#a = torch.ones(l).to(inp.device)
		#a[-1] = self.alpha
		#b = torch.zeros(l).to(inp.device)
		#b[-1] = self.beta
		
		#tmp = F.softmax(inp, 1) * a.view(1,-1,1,1) + b.view(1,-1,1,1)
		#return F.normalize(tmp, 1)
		#return inp * a.view(1,-1,1,1) + b.view(1,-1,1,1)
		
		
		#tmp = F.softmax(inp,1) * self.alpha.view(1,-1,1,1)
		#return F.normalize(tmp, 1)

		return inp * self.alpha.view(1,-1,1,1) + self.beta.view(1,-1,1,1)



	def loss_fn(self, predicted, labels):

		#return criterion(predicted, F.normalize(labels,1))
		return criterion(predicted, labels.argmax(dim=1, keepdim=False))
		
		"""
		n,l,h,w = labels.shape
		d = labels.device
		predicted = torch.clamp(torch.floor(predicted-predicted.max(dim=1, keepdim=True).values + torch.tensor(1.).to(predicted.device)), 0.0, 1.0)
		labels = torch.clamp(torch.floor(labels-labels.max(dim=1, keepdim=True).values + torch.tensor(1.).to(labels.device)), 0.0, 1.0)
		

		if l==2:
			TP,FP,TN,FN = torch.tensor(0.,device=d), torch.tensor(0., device=d), torch.tensor(0., device=d), torch.tensor(0., device=d)
			TP+= (predicted[:,0,:,:] * labels[:,0,:,:]).sum()
			FP+= (predicted[:,1,:,:] * labels[:,0,:,:]).sum()
			TN+= (predicted[:,1,:,:] * labels[:,1,:,:]).sum()
			FN+= (predicted[:,0,:,:] * labels[:,1,:,:]).sum()

			p = TP / (TP + FP)
			r = TP/ (TP + FN)
			F1 = 2* p*r / (p+r)
			return torch.tensor(1.,device=d)-F1

		else: #l=4
			precision = torch.tensor([], device=d)
			recall = torch.tensor([], device=d)

			TP,FP,TN,FN = torch.tensor(0.,device=d), torch.tensor(0., device=d), torch.tensor(0., device=d), torch.tensor(0., device=d)
			ground = torch.cat( [labels.index_select(1, torch.tensor([0]).to(device)), labels.index_select(1, torch.tensor([1,2,3]).to(device)).sum(1, keepdim=True)], 1)
			pred = torch.cat( [predicted.index_select(1, torch.tensor([0]).to(device)), labels.index_select(1, torch.tensor([1,2,3]).to(device)).sum(1, keepdim=True)], 1)
			TP+= (ground[:,0,:,:] * pred[:,0,:,:]).sum()
			FP+= (ground[:,1,:,:] * pred[:,0,:,:]).sum()
			TN+= (ground[:,1,:,:] * pred[:,1,:,:]).sum()
			FN+= (ground[:,0,:,:] * pred[:,1,:,:]).sum()
			p = TP / (TP + FP)
			r = TP/ (TP + FN)
			precision=torch.cat([precision, p.view(1)])
			recall=torch.cat([recall, r.view(1)])

			TP,FP,TN,FN = torch.tensor(0.,device=d), torch.tensor(0., device=d), torch.tensor(0., device=d), torch.tensor(0., device=d)
			ground = torch.cat( [labels.index_select(1, torch.tensor([1]).to(device)), labels.index_select(1, torch.tensor([0,2,3]).to(device)).sum(1, keepdim=True)], 1)
			pred = torch.cat( [predicted.index_select(1, torch.tensor([1]).to(device)), labels.index_select(1, torch.tensor([0,2,3]).to(device)).sum(1, keepdim=True)], 1)
			TP+= (ground[:,0,:,:] * pred[:,0,:,:]).sum()
			FP+= (ground[:,1,:,:] * pred[:,0,:,:]).sum()
			TN+= (ground[:,1,:,:] * pred[:,1,:,:]).sum()
			FN+= (ground[:,0,:,:] * pred[:,1,:,:]).sum()
			p = TP / (TP + FP)
			r = TP/ (TP + FN)
			precision=torch.cat([precision, p.view(1)])
			recall=torch.cat([recall, r.view(1)])

			TP,FP,TN,FN = torch.tensor(0.,device=d), torch.tensor(0., device=d), torch.tensor(0., device=d), torch.tensor(0., device=d)
			ground = torch.cat( [labels.index_select(1, torch.tensor([2]).to(device)), labels.index_select(1, torch.tensor([0,1,3]).to(device)).sum(1, keepdim=True)], 1)
			pred = torch.cat( [predicted.index_select(1, torch.tensor([2]).to(device)), labels.index_select(1, torch.tensor([0,1,3]).to(device)).sum(1, keepdim=True)], 1)
			TP+= (ground[:,0,:,:] * pred[:,0,:,:]).sum()
			FP+= (ground[:,1,:,:] * pred[:,0,:,:]).sum()
			TN+= (ground[:,1,:,:] * pred[:,1,:,:]).sum()
			FN+= (ground[:,0,:,:] * pred[:,1,:,:]).sum()
			p = TP / (TP + FP)
			r = TP/ (TP + FN)
			precision=torch.cat([precision, p.view(1)])
			recall=torch.cat([recall, r.view(1)])

			print("p,r", precision, recall)
			pn = precision.mean()
			rl = recall.mean()
			F1 = 2.* pn*rl / (pn+rl)
			return torch.tensor(1.,device=d)-F1
			"""



	def fit(self, observations, labels):
		def closure():
			predicted = self.forward(observations)
			loss = self.loss_fn(predicted, labels)
			#print("test: ", loss)
			self.optimizer.zero_grad()
			loss.backward(retain_graph=True)
			return loss
		old_params = parameters_to_vector(self.parameters())
		for lr in LR * .5**np.arange(10):
			self.optimizer = optim.LBFGS(self.parameters(), lr=lr)
			self.optimizer.step(closure)
			current_params = parameters_to_vector(self.parameters())
			if any(np.isnan(current_params.data.cpu().numpy())):
				print("LBFGS optimization diverged. Rolling back update...")
				vector_to_parameters(old_params, self.parameters())
			else:
				return

def main():

	valid_datasets = {}


	## Validation set
	valid_datasets['eyebrow'] = ConcatDataset([make_dataset('tuning.txt', 'eyebrow1', fg_indexs=set([2]), trans=[ToTensor()]), 
											   make_dataset('tuning.txt', 'eyebrow2', fg_indexs=set([3]), trans=[ToTensor(), Invert()])])

	valid_datasets['eye'] = ConcatDataset([make_dataset('tuning.txt', 'eye1', fg_indexs=set([4]), trans=[ToTensor()]), 
										make_dataset('tuning.txt', 'eye2', fg_indexs=set([5]), trans=[ToTensor(), Invert()])])

	valid_datasets['nose'] = make_dataset('tuning.txt', 'nose', fg_indexs=set([6]), trans=[ToTensor()])
	valid_datasets['mouth'] = make_dataset('tuning.txt', 'mouth', fg_indexs=set([7,8,9]), trans=[ToTensor()])


	names = ['eyebrow', 'eye', 'nose', 'mouth']
	l = {'eyebrow':2, 'eye':2, 'nose':2, 'mouth':4}

	for name in names:
		print(name, "...")
		model = pickle.load(open('res/saved-model-%s.pth'%name, 'rb'))
		#model model.to(device)

		mod = Modulator(l[name])
		mod = mod.to(device)

		valid_loader = DataLoader(valid_datasets[name], batch_size=230, shuffle=True, num_workers=4)

		batch = [b for b in valid_loader][0]
		image, labels = batch['image'].to(device), batch['labels'].to(device)
		observations = model(image)
		loss_before = mod.loss_fn(observations, labels)
		mod.fit(observations, labels)
		loss_after = mod.loss_fn(mod.forward(observations), labels)
		print("Loss Before", loss_before)
		print("Loss After", loss_after)

		pickle.dump(mod, open('res/saved-modulator-%s.pth'%name, 'wb'))

if __name__ == '__main__':
	main()