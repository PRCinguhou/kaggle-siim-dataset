import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from LoadData import siim_dataset
from model import U_resnet, Bottleneck, UnetBlock
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
import torch.nn.functional as F
import random
import torch.optim as optim
from tqdm import tqdm
import segmentation_models_pytorch as smp

parser = ArgumentParser()
parser.add_argument("-EPOCH", "--EPOCH", dest="epoch", type=int, default=200)
parser.add_argument("-batch_size", dest='batch_size', type=int, default=30)
parser.add_argument("-model", dest='model', type=str, default='simm-dataset_Unet')
parser.add_argument("-lr", dest='lr', type=float, default=1e-3)
args = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

def dice_coef(y_true, y_pred):
	smooth = 1
	y_true_f = y_true.flatten()
	y_pred_f = y_pred.flatten()

	intersection = torch.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_loss_fn(y_true, y_pred):
	return 1-dice_coef(y_true, y_pred)

def IOU(pred, label):
	total_score, valid_count = 0, 0
	for x, y in zip(pred, label):
		if torch.sum(y) == 0:
			continue
		smooth = 1
		pred_f = x.flatten()
		pred_f = (pred_f>=0.5) * 1
		label_f = y.flatten()
		intersection = torch.sum(pred_f.int() * label_f.int())
		union = torch.sum(pred_f) + torch.sum(label_f) - intersection
		
		total_score +=  intersection/(union + smooth)
		valid_count += 1
		
	return total_score / (valid_count+1)

def go(x):
	return x

if __name__ == '__main__':
	
	#model = U_resnet(Bottleneck, UnetBlock, [1,1,1,1]).to(device)
	model = smp.Unet("resnet18", encoder_weights="imagenet", activation=go).to(device)
	
	dataset = siim_dataset('dicom-images-train', 'train-rle.csv')
	dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle = True)
	
	#model.load_state_dict(torch.load('./model_100.pth'))
	
	loss = smp.utils.losses.BCEDiceLoss(eps=1.)
	

	optimizer = optim.Adam([
		{'params': model.decoder.parameters(), 'lr': args.lr},
		{'params': model.encoder.parameters(), 'lr': args.lr},])

	
	print(f"""
		Current Hyper-Parameters:
		o Model Type: {args.model},
		o Epoch : {args.epoch},
		o Batch Size : {args.batch_size},
		o Learning Rate : {args.lr},
		""")

	metrics = [ smp.utils.metrics.IoUMetric(eps=1.), smp.utils.metrics.FscoreMetric(eps=1.),]

	train_epoch = smp.utils.train.TrainEpoch(
		model, 
		loss=loss, 
		metrics=metrics,
		optimizer=optimizer,
		device=device,
		verbose=True,
	)

	max_score = 0
	for i in range(0, 40):
		print('\nEpoch: {}'.format(i))

		train_logs = train_epoch.run(dataset)
		
		# do something (save model, change lr, etc.)
		if max_score < train_logs['iou']:
			max_score = train_logs['iou']
			torch.save(model, './best_model.pth')
			print('Model saved!')
			
		if i == 25:
			optimizer.param_groups[0]['lr'] = 1e-5
			print('Decrease decoder learning rate to 1e-5!')
	# for ep in tqdm(range(args.epoch)):
	#   avg_bce_loss = 0
	#   avg_iou = 0
	#   for index, batch in enumerate(dataset):
		
	#       x, y = batch
	#       x = x.to(device)
	#       y = y.to(device)
			
	#       output = model(x)
	#       # print(output)
	#       bce_loss = loss_fn(output, y)

	#       iou = IOU(output, y)
	#       try:
	#           avg_iou += iou.item()
	#       except:
	#           pass
	#       loss = bce_loss
			
	#       avg_bce_loss += bce_loss.item()
	#       optimizer.zero_grad()
	#       loss.backward()
	#       optimizer.step()
	#       #test = output[0][0].cpu().detach()

	#   print('Avg BCE Loss : [%.4f], Avg IOU : [%.4f]' % (avg_bce_loss/len(dataset), (avg_iou/len(dataset))))      
		
	#   if ep < 100:
	#       torch.save(model.state_dict(), './model_100.pth')
	#   elif ep < 200:
	#       torch.save(model.state_dict(), './model_200.pth')
	#   elif ep < 300:
	#       torch.save(model.state_dict(), './model_300.pth')
	#   elif ep < 400:
	#       torch.save(model.state_dict(), './model_400.pth')
	#   else:
	#       torch.save(model.state_dict(), './model_500.pth')