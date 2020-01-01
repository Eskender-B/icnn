import cv2
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import shutil
from mtcnn.mtcnn import MTCNN
from skimage import io, transform


def prepare(root_dir, txt_file):

	name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype='str', delimiter=',')
	lmarks_list = []

	for idx in range(name_list.shape[0]):
		img_name = os.path.join(root_dir, 'images',
					name_list[idx, 1].strip() + '.jpg')

		image = io.imread(img_name)

		
		detector = MTCNN()
		landmarks = detector.detect_faces(image)[0]['keypoints']
		landmarks = np.array([landmarks[key] for key in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']])
		
		lmarks_list.append(landmarks.reshape(-1))
		print(txt_file + ' :', idx)


	name_lmarks_list = np.concatenate((name_list, lmarks_list), axis=1)
	np.savetxt(os.path.join(root_dir, txt_file), name_lmarks_list, fmt='%s', delimiter=',')


root_dir='data/SmithCVPR2013_dataset_resized'
for file in ['testing.txt', 'tuning.txt', 'exemplars.txt']:
	prepare(root_dir, file)