from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os 
from os.path import join
from os import listdir
import pandas as pd
from mask_functions import read_dicom, mask2rle, rle2mask

mean = [0.5]
std = [0.5]

transform = transforms.Compose([
	transforms.ToTensor(),
	# transforms.Normalize(mean, std)
	])

class siim_dataset(Dataset):

	def __init__(self, img_path, label_csv):
		
		self.img_path = img_path
		self.label_csv = label_csv
		self.imgs_dir = listdir(join(os.getcwd(), self.img_path))
		###############################################################
		######		 			  [csv format]		  			 ######
		######		ImageId						EncodedPixels	 ######
		###############################################################
		self.label_csv = pd.read_csv(join(os.getcwd(), self.label_csv))
		self.image_id = self.label_csv['ImageId'].values
		self.image_mask = self.label_csv[' EncodedPixels'].values


	def __len__(self):
		return len(self.imgs_dir)

	def __getitem__(self, idx):

		dir_name = self.imgs_dir[idx]
		sub_dir_name = listdir(join(os.getcwd(), self.img_path, dir_name))[0]
		image_name = listdir(join(os.getcwd(), self.img_path, dir_name, sub_dir_name))[0]


		# image_mask = self.image_mask[idx]

		image_mask = self.label_csv[self.label_csv['ImageId'] == image_name.replace('.dcm', '')][' EncodedPixels'].values[0]
		image, width, height = read_dicom(join(os.getcwd(), self.img_path, dir_name, sub_dir_name, image_name))
		label = rle2mask(image_mask, width, height)
		image = transform(image)
		label = transform(label)

		return image, label

# if __name__ == '__main__':
# 	dataset = siim_dataset('dicom-images-train', 'train-rle.csv')
# 	dataset = DataLoader(dataset, batch_size=1)
# 	for index, i in enumerate(dataset):
# 		x, y = i

# 		if index == 100:
# 			break