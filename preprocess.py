from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ImageDataset(Dataset):
	"""Image dataset."""
	def __init__(self, txt_file, root_dir, transform=None):
		"""
		Args:
		txt_file (string): Path to the txt file with list of image id, name.
		root_dir (string): Directory with all the images.
		transform (callable, optional): Optional transform to be applied
		on a sample.
		"""
		self.name_list = np.loadtxt(txt_file, dtype='str', delimiter=',')
		self.root_dir = root_dir
		self.transform = transform
	def __len__(self):
		return len(self.name_list)
	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, 'images',
		self.name_list[idx, 1].strip() + '.jpg')
		image = io.imread(img_name)
		label_name = os.path.join(self.root_dir, 'labels',
		self.name_list[idx, 1].strip(), self.name_list[idx, 1].strip() + '_lbl%.2d.png')
		labels = []
		for i in range(1,11):
			labels.append(io.imread(label_name%i))
		labels = np.array(labels, dtype=np.float)
		sample = {'image': image, 'labels': labels}
		if self.transform:
			sample = self.transform(sample)
		return sample