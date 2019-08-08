import torch.nn as nn
import torch




class Bottleneck(nn.Module):

	expansion = 4

	def __init__(self, in_channel, out_channel, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channel)
		self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channel)
		self.conv3 = nn.Conv2d(out_channel, out_channel*4, 1, 1, 0, bias=False)
		self.bn3 = nn.BatchNorm2d(out_channel*4)
		self.relu = nn.ReLU(True)

		self.stride = stride
		self.downsample = downsample

	def forward(self, x):
		residual = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class UnetBlock(nn.Module):

	def __init__(self, in_channel, out_channel):
		super(UnetBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channel, out_channel, 5, 1, 2)
		if out_channel != 1:
			self.bn1 = nn.BatchNorm2d(out_channel)
		else:
			self.bn1 = None
		self.relu = nn.ReLU(True)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.sig = nn.Sigmoid()

	def forward(self, x):

		out = self.conv1(x)
		out = self.relu(out)
		if self.bn1 is not None:
			out = self.bn1(out)
			out = self.upsample(out)
		else:
			out = self.sig(out)

		return out 


class U_resnet(nn.Module):

	def __init__(self, block, unetblock, layers, num_class=2):
		super(U_resnet, self).__init__()

		##### Resnet Elemnet #####
		self.in_channel = 8
		self.conv1 = nn.Conv2d(1, 8, 5, 2, 2)
		self.bn1 = nn.BatchNorm2d(8)
		self.relu = nn.ReLU(True)
		# self.maxpool = nn.MaxPool2d(2)
		self.layer1 = self._make_layer(block, 8, layers[0], 2)
		self.layer2 = self._make_layer(block, 16, layers[1], 2)
		self.layer3 = self._make_layer(block, 64, layers[2], 2)
		self.layer4 = self._make_layer(block, 128, layers[3], 2)

		##### Unet #####
		self.up1 = unetblock(512, 256)
		self.up2 = unetblock(512, 64)
		self.up3 = unetblock(128, 32)
		self.up4 = unetblock(64, 8)
		self.up5 = unetblock(16, 1)
		self.final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

	def _make_layer(self, block, out_channel, blocks, stride=1):
		downsample = None

		if stride != 1 or self.in_channel != out_channel * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.in_channel, out_channel * block.expansion, 1, stride, 0),
				nn.BatchNorm2d(out_channel * block.expansion)
				)
		layers = []
		layers.append(block(self.in_channel, out_channel, stride, downsample))
		self.in_channel = out_channel * block.expansion

		for i in range(1, blocks):
			layers.append(block(self.in_channel, out_channel))

		return nn.Sequential(*layers)

	def forward(self, x):
		# 1 x 512 x 512
		out1 = self.conv1(x)

		# 8 x 256 x 256
		out1 = self.bn1(out1)
		out1 = self.relu(out1)
		# out = self.maxpool(out)


		out2 = self.layer1(out1)
		# 32 x 256 x 256

		out3 = self.layer2(out2)
		# 64 x 128 x 128

		out4 = self.layer3(out3)
		# 256 x 64 x 64

		out5 = self.layer4(out4)
		# 512 x 32 x 32
		
		# x    : 1 x 512 x 512
		# out1 : 8 x 256 x 256
		# out2 : 32 x 128 x 128
		# out3 : 64 x 64 x64
		# out4 : 256 x 32 x 32
		# out5 : 512 x 16 x16
		
		up1 = self.up1(out5)
		up1 = torch.cat([out4, up1], dim=1)

		up2 = self.up2(up1)
		up2 = torch.cat([out3, up2], dim=1)

		up3 = self.up3(up2)
		up3 = torch.cat([out2, up3], dim=1)

		up4 = self.up4(up3)
		up4 = torch.cat([out1, up4], dim=1)

		up5 = self.up5(up4)
		up5 = self.final(up5)

		return up5


# model = U_resnet(Bottleneck, UnetBlock, [10,10,10,10])
# x = torch.ones((1, 1, 512, 512))
# model(x)











