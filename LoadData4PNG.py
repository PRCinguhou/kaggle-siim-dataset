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
	transforms.Resize((256,256)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])

label_transform = transforms.Compose([
	transforms.Resize((256, 256)),
	transforms.ToTensor()
	])


class dataloader(Dataset):

	def __init__(self, img_path, label_path):

		self.imgs = sorted(listdir(join(os.getvwd(), img_path)))
		self.labels = sorted(listdir(join(os.getvwd(), label_path)))
		self.img_path = img_path
		self.label_path = label_path

	def __len__(self):
		return len(self.imgs)


	def __getitem__(self, idx):

		print(self.imgs[idx])
		print(self.labels[idx])

		image = Image.open(join(os.getcwd(), self.img_path, self.imgs[idx]))
		label = Image.open(join(os.getcwd(), self.label_path, self.labels[idx]))

		image = transform(image)
		label = label_transform(label)

		return image, label