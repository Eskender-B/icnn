from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Rescale(object):
	"""Rescale the image in a sample to a given size.

	Args:
	output_size (tuple or int): Desired output size. If tuple, output is
	matched to output_size. If int, smaller of image edges is matched
	to output_size keeping aspect ratio the same.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, labels, idx = sample['image'], sample['labels'], sample['index']

		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		new_img = transform.resize(image, (new_h, new_w))
		new_labels = np.zeros([labels.shape[0], new_h, new_w], dtype=labels.dtype)

		for i in range(labels.shape[0]):
			new_labels[i,:,:] = transform.resize(labels[i,:,:], (new_h, new_w))

		return {'image': new_img, 'labels': new_labels, 'index', idx}


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, labels, idx = sample['image'], sample['labels'], sample['index']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image).float(),
		'labels': torch.from_numpy(labels).float(),
		'index': idx}




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
		self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype='str', delimiter=',')
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
		for i in range(11):
			labels.append(io.imread(label_name%i))
		labels = np.array(labels, dtype=np.float)

		# calculate background pixels (hair & skin & actual backround)
		bg = labels[0]+labels[1]+labels[10]
		labels = np.concatenate((labels[2:10] ,[bg.clip(0.0,255.0)]), axis=0) 
		
		sample = {'image': image, 'labels': labels, 'index':idx}
		if self.transform:
			sample = self.transform(sample)
		return sample