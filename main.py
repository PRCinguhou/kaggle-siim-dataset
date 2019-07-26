import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from LoadData import siim_dataset
from model import Unet
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
import torch.nn.functional as F
import random
import torch.optim as optim
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-EPOCH", "--EPOCH", dest="epoch", type=int, default=200)
parser.add_argument("-batch_size", dest='batch_size', type=int, default=30)
parser.add_argument("-model", dest='model', type=str, default='simm-dataset_Unet')
parser.add_argument("-lr", dest='lr', type=float, default=1e-3)
args = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

if __name__ == '__main__':

	model = Unet().to(device)
	# model.load_state_dict(torch.load('./model_1.pth'))
	loss_fn = nn.MSELoss()
	dataset = siim_dataset('dicom-images-train', 'train-rle.csv')
	dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle = True)
	optimizer = optim.Adam(model.parameters(), lr = args.lr)
	print(f"""
		Current Hyper-Parameters:
		o Model Type: {args.model},
		o Epoch : {args.epoch},
		o Batch Size : {args.batch_size},
		o Learning Rate : {args.lr},
		""")
	for ep in tqdm(range(args.epoch)):
		avg_loss = 0
		for index, batch in enumerate(dataset):
			
			x, y = batch
			x = x.to(device)
			y = y.to(device)
			
			output = model(x)
			
			loss = loss_fn(output, y)
			avg_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			test = output[0][0].cpu().detach()
		print('Avg Loss : [%.4f]' % (avg_loss/len(dataset)))		
		if ep == 199:
			plt.imshow(test, cmap='gray')
			plt.show()

		torch.save(model.state_dict(), './model.pth')