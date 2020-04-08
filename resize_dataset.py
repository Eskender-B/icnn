import cv2
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import shutil
from mtcnn.mtcnn import MTCNN
from skimage import io, transform

resize_num = 64

def prepare(root_dir, new_dir, txt_file):

	name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype='str', delimiter=',')
	lmarks_list = []

	for idx in range(name_list.shape[0]):
		img_name = os.path.join(root_dir, 'images',	name_list[idx, 1].strip() + '.jpg')
		image = io.imread(img_name)

		label_name = os.path.join(root_dir, 'labels', name_list[idx, 1].strip(), name_list[idx, 1].strip() + '_lbl%.2d.png')
		labels = []
		for i in range(11):
			labels.append(io.imread(label_name%i))
		labels = np.array(labels, dtype=np.uint8)

		# Resize
		image = np.uint8(transform.resize(image, (resize_num, resize_num)) * 255)
		labels = transform.resize(labels.transpose(1,2,0), (resize_num,resize_num)).transpose(2,0,1)
		labels = np.uint8((labels/labels.sum(axis=0, keepdims=True))*255)

		img_name = os.path.join(new_dir, 'images', name_list[idx, 1].strip() + '.jpg')
		io.imsave(img_name, image, quality=100)

		shutil.os.mkdir(shutil.os.path.join(new_dir, 'labels', name_list[idx, 1].strip()))
		label_name = shutil.os.path.join(new_dir, 'labels', name_list[idx, 1].strip(), name_list[idx, 1].strip() + '_lbl%.2d.png')
		for i in range(len(labels)):
			io.imsave(label_name%i, labels[i], check_contrast=False)

		print(txt_file + ' :', idx)



	np.savetxt(os.path.join(new_dir, txt_file), name_list, fmt='%s', delimiter=',')


root_dir='data/SmithCVPR2013_dataset_resized'
new_dir='data/SmithCVPR2013_dataset_resized_' + str(resize_num)

# Clean first
if shutil.os.path.exists(new_dir):
  shutil.rmtree(new_dir)
shutil.os.mkdir(new_dir)

shutil.os.mkdir(new_dir+'/images')
shutil.os.mkdir(new_dir+'/labels')

for file in ['testing.txt', 'tuning.txt', 'exemplars.txt']:
	prepare(root_dir, new_dir, file)