import numpy as np
import torch.nn as nn
import torch
from model import Unet
from os.path import join
import os 
from os import listdir
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from mask_functions import read_dicom, mask2rle, rle2mask
import pandas as pd

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')


parser = ArgumentParser()
parser.add_argument("-img", "--img", dest="img_index", type=int, default=100)
args = parser.parse_args()

std = [0.5]
mean = [0.5]
transform = transforms.Compose([
	# transforms.Grayscale(),
	transforms.ToPILImage(),
	transforms.Resize((512,512)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])
label_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize((512, 512)),
	transforms.ToTensor()
	])
def IOU(pred, label):
	total_score, valid_count = 0, 0
	for x, y in zip(pred, label):
		y = y.numpy()
		if np.sum(y) == 0:
			continue
		smooth = 1
		pred_f = x.flatten().astype(float)
		label_f = y.flatten()
		intersection = np.sum(pred_f * label_f)
		union = np.sum((pred_f.astype(int) | label_f.astype(int)))
		total_score +=  intersection/(union + smooth)
		valid_count += 1
		
	return total_score / valid_count
	
def xchest(image_path, image_name):
	model = Unet().to(device)
	model.load_state_dict(torch.load('./model_100.pth'))
	model.eval()

	image, w, h = read_dicom(image_path)
	image = transform(image).to(device)
	image = image.view(1, 1, 512, 512)
	res = model(image.float())[0].cpu().detach().numpy()[0]

	res =( res>=0.4)*1
	

	csv_data = pd.read_csv('./train-rle.csv')
	table = {}
	for x, y in zip(csv_data['ImageId'].values, csv_data[' EncodedPixels'].values):
		table[x] = y
	
	mask_rle = table[image_name]
	if mask_rle ==  ' -1':
		return 

	mask_image = rle2mask(mask_rle, w, h)
	mask_image = label_transform(mask_image)[0]
	
	print(IOU([res], [mask_image.t()]))
	plt.figure(1)
	plt.subplot(211)
	plt.title('res')
	plt.imshow(image.cpu().detach().numpy().reshape(512,512))
	plt.imshow((mask_image*255).numpy().T, alpha=0.5, cmap='gray')
	plt.subplot(212)
	plt.imshow(res*255, cmap='gray')
	plt.show()

for i in range(1000):

	files = listdir(join(os.getcwd(), 'dicom-images-train'))[i]
	sub_file = listdir(join(os.getcwd(), 'dicom-images-train', files))[0]
	image_name = listdir(join(os.getcwd(), 'dicom-images-train', files, sub_file))[0]
	xchest(join(os.getcwd(), 'dicom-images-train', files, sub_file, image_name), image_name.replace('.dcm', ''))
