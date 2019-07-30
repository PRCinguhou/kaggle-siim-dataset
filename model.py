import torch
import torch.nn as nn

class Unet(nn.Module):

	def __init__(self):
		super(Unet, self).__init__()

		# 512 x 512
		self.encoder1 = nn.Sequential(
			nn.Conv2d(1, 16, 5, 2, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.Dropout(0.1)
			)

		# 256 x 256
		self.encoder2 = nn.Sequential(
			nn.Conv2d(16, 32, 5, 2, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.Dropout(0.1)
			)

		# 128 x 128
		self.encoder3 = nn.Sequential(
			nn.Conv2d(32, 64, 5, 2, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.Dropout(0.1)
			)

		# 64 x 64
		self.encoder4 = nn.Sequential(
			nn.Conv2d(64, 128, 5, 2, 2),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.Dropout(0.1)
			)


		self.f64t16 = nn.Sequential(
			nn.Conv2d(64, 16,3, 1, 1),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.Dropout(0.1)
			)

		self.f128t32 = nn.Sequential(
			nn.Conv2d(128, 32 ,3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.Dropout(0.1)
			)

		self.f128t64 = nn.Sequential(
			nn.Conv2d(128, 64 ,3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.Dropout(0.1)
			)


		self.final = nn.Sequential(
			nn.Conv2d(32, 16, 3, 1, 1),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.Conv2d(16, 1, 3, 1, 1),
			nn.Sigmoid()
			)


	def forward(self, img):
		upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		# 8 x 256 x 256
		e1 = self.encoder1(img)
		# 16 x 128 x 128
		e2 = self.encoder2(e1)
		# 32 x 64 x 64
		e3 = self.encoder3(e2)
		# 64 x 32 x 32
		e4 = self.encoder4(e3)

		
		e9 = upsample(e4)
		# 64 x 128 x 128
		e9 = self.f128t64(e9)
		# 16 x 128 x 128
		e9 = torch.cat([e3, e9], dim=1)
		# 32 x 128 x 128

		e10 = upsample(e9)
		# 64 x 128 x 128
		e10 = self.f128t32(e10)
		# 16 x 128 x 128
		e10 = torch.cat([e2, e10], dim=1)
		# 32 x 128 x 128

		e11 = upsample(e10)
		# 32 x 256 x 256
		e11 = self.f64t16(e11)
		# 8 x 256 x 256
		e11 = torch.cat([e1, e11], dim=1)
		# 16 x 256 x 256

		e11 = upsample(e11)
		result = self.final(e11)

		return result
