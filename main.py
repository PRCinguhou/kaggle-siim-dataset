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

def dice_loss_fn(pred, label):

	smooth = 1
	pred_f = pred.flatten()
	label_f = label.flatten()
	intersection = pred_f * label_f
	score = (2 * torch.sum(intersection) + smooth) / (torch.sum(pred_f) + torch.sum(label_f) + smooth)
	return 1 - score
def IOU(pred, label):
	total_score, valid_count = 0, 0
	for x, y in zip(pred, label):
		if torch.sum(y) == 0:
			continue
		smooth = 1
		pred_f = x.flatten().float()
		label_f = y.flatten()
		intersection = torch.sum(pred_f * label_f)
		union = torch.sum((pred_f.int() | label_f.int()))
		total_score +=  intersection/(union + smooth)
		valid_count += 1
		
	return total_score / (valid_count+1)
	
if __name__ == '__main__':
	
	model = Unet().to(device)
	model.load_state_dict(torch.load('./model_100.pth'))
	loss_fn = nn.BCELoss()
	dataset = siim_dataset('dicom-images-train', 'train-rle.csv')
	dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle = True)
	optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay=0.9)
	print(f"""
		Current Hyper-Parameters:
		o Model Type: {args.model},
		o Epoch : {args.epoch},
		o Batch Size : {args.batch_size},
		o Learning Rate : {args.lr},
		""")
	for ep in tqdm(range(args.epoch)):
		avg_bce_loss = 0
		avg_dice_loss = 0
		avg_iou = 0
		avg_healthy = 0
		not_health_count = 0
		for index, batch in enumerate(dataset):
			
			x, y = batch
			x = x.to(device)
			y = y.to(device)
			
			output = model(x)
			bce_loss = loss_fn(output, y)
			dice_loss = dice_loss_fn(output, y)
			iou = IOU(output, y)
			try:
				avg_iou += iou.item()
			except:
				pass
			loss = bce_loss + dice_loss
			#print(loss)
			avg_bce_loss += bce_loss.item()
			avg_dice_loss += dice_loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#test = output[0][0].cpu().detach()
		print('Avg BCE Loss : [%.4f], Avg Dice Loss : [%.4f], Avg IOU : [%.4f]' % (avg_bce_loss/len(dataset), avg_dice_loss/len(dataset), (avg_iou/len(dataset))))		
		
		if ep < 100:
			torch.save(model.state_dict(), './model_100.pth')
		elif ep < 200:
			torch.save(model.state_dict(), './model_200.pth')
		elif ep < 300:
			torch.save(model.state_dict(), './model_300.pth')
		elif ep < 400:
			torch.save(model.state_dict(), './model_400.pth')
		else:
			torch.save(model.state_dict(), './model_500.pth')