import torch
import torch.nn as nn

class Unet(nn.Module):

	def __init__(self):
		super(refine_net, self).__init__()

		# 512 x 512
		self.encoder1 = nn.Sequential(
			nn.Conv2d(1, 8, 5, 2, 2),
			nn.BatchNorm2d(8),
			nn.ReLU(True),
			)

		# 256 x 256
		self.encoder2 = nn.Sequential(
			nn.Conv2d(8, 16, 5, 2, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			)

		# 128 x 128
		self.encoder3 = nn.Sequential(
			nn.Conv2d(16, 32, 5, 2, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			)

		# 64 x 64
		self.encoder4 = nn.Sequential(
			nn.Conv2d(32, 64, 5, 2, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			)

		# 32 x 32
		self.encoder5 = nn.Sequential(
			nn.Conv2d(64, 128, 5, 2, 2),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			)

		self.encoder6 = nn.Sequential(
			nn.Conv2d(128, 256, 5, 2, 2),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			)

		self.f256t128 = nn.Sequential(
			nn.Conv2d(256, 128 ,3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(True)
			)

		self.f256t64 = nn.Sequential(
			nn.Conv2d(128, 64 ,3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(True)
			)

		self.f128t32 = nn.Sequential(
			nn.Conv2d(32, 16 ,3, 1, 1),
			nn.BatchNorm2d(16),
			nn.ReLU(True)
			)

		self.f64t16 = nn.Sequential(
			nn.Conv2d(16, 8 ,3, 1, 1),
			nn.BatchNorm2d(8),
			nn.ReLU(True)
			)

		self.f32t8 = nn.Sequential(
			nn.Conv2d(32, 8 ,3, 1, 1),
			nn.BatchNorm2d(8),
			nn.ReLU(True)
			)

		self.final = nn.Sequential(
			nn.Conv2d(16, 1, 5, 1, 2),
			nn.Sigmoid()
			)


	def forward(self, img):
		upsample = nn.Upsample(size=2, mode='bilinear', align_corners=True)
		# 8 x 256 x 256
		e1 = self.encoder1(img)
		# 16 x 128 x 128
		e2 = self.encoder2(e1)
		# 32 x 64 x 64
		e3 = self.encoder3(e2)
		# 64 x 32 x 32
		e4 = self.encoder4(e3)
		# 128 x 16 x 16
		e5 = self.encoder5(e4)

		e6 = self.encoder6(e5)

		e7 = upsample(e5)
		# 128 x 32 x32
		e7 = self.f256t128(e6)
		# 64 x 32 x32
		e7 = torch.cat([e5, e7], dim=1)
		# 128 x 32 x32


		e8 = upsample(e7)
		# 128 x 64 x 64
		e8 = self.f256t64(e7)
		# 32 x 64 x 64
		e8 = torch.cat([e4, e8], dim=1)
		# 64 x 64 x 64

		e9 = upsample(e8)
		# 64 x 128 x 128
		e9 = self.f128t16(e8)
		# 16 x 128 x 128
		e9 = torch.cat([e2, e8], dim=1)
		# 32 x 128 x 128

		e10 = upsample(e9)
		# 32 x 256 x 256
		e10 = self.f32t8(e9)
		# 8 x 256 x 256
		e10 = torch.cat([e1, e9], dim=1)
		# 16 x 256 x 256

		result = self.final(e10)


		return result
