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
from collections import Counter
import torch.nn as nn
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize((256,256)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])

label_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize((256, 256)),
	transforms.ToTensor()
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
		self.id2mask = {}
		for x, y in zip(self.image_id, self.image_mask):
			self.id2mask[x]=y

	def __len__(self):
		return len(self.imgs_dir)

	def __getitem__(self, idx):

		dir_name = self.imgs_dir[idx]
		sub_dir_name = [file for file in listdir(join(os.getcwd(), self.img_path, dir_name)) if file != '.DS_Store'][0]
		image_name = [file for file in listdir(join(os.getcwd(), self.img_path, dir_name, sub_dir_name)) if file != '.DS_Store'][0]

		#image_mask = self.label_csv[self.label_csv['ImageId'] == image_name.replace('.dcm', '')][' EncodedPixels'].values[0]
		image_mask = self.id2mask[image_name.replace('.dcm','')]

		image, width, height = read_dicom(join(os.getcwd(), self.img_path, dir_name, sub_dir_name, image_name))
		
		label = rle2mask(image_mask, width, height).T
	
		image = transform(image)
		label = label_transform(label)
		label = label.view(1, 256, 256)

		return image, label

if __name__ == '__main__':
	dataset = siim_dataset('dicom-images-train', 'train-rle.csv')
	dataset = DataLoader(dataset, batch_size=1)
	fn = nn.BCELoss()
	loss = 0
	for index, i in enumerate(dataset):
		x, y = i
		out = torch.ones(x.size())
		out /= 2
		c_loss = fn(out, y)
		print(c_loss)
		loss += c_loss.item()
	print(loss/len(dataset))