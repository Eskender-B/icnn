from utils import shift
import torch
import torch.nn.functional as F
import torch.nn as nn


class ICNN(nn.Module):

	def __init__(self):
		super(ICNN,self).__init__()
		self.num_rows = 4
		self.num_interlink_layer = 3
		self.scaling_factor = 2
		self.kernel_size = 5
		self.last_kernel_size = 9
		self.L = 9
		self.num_channel_orignal = [8*i for i in range(1, self.num_rows+1)]		# [8, 16, 24, 32]
		self.num_channel_interlinked = shift(num_channel_orignal, -1, 0) + num_channel_orignal + shift(num_channel_orignal, 1, 0) 

		# Input convs
		self.inp_convs = nn.ModuleList([nn.Conv2D(num_channel_orignal[r], 3, self.kernel_size, padding=self.kernel_size//2) for r in range(self.num_rows)])


		# Interlinking convs
		self.inter_convs_row0 = nn.ModuleList([nn.Conv2D(num_channel_orignal[0], num_channel_interlinked[0], self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])
		self.inter_convs_row1 = nn.ModuleList([nn.Conv2D(num_channel_orignal[1], num_channel_interlinked[1], self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])
		self.inter_convs_row2 = nn.ModuleList([nn.Conv2D(num_channel_orignal[2], num_channel_interlinked[2], self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])
		self.inter_convs_row3 = nn.ModuleList([nn.Conv2D(num_channel_orignal[3], num_channel_interlinked[3], self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])


		# Output convs
		self.out_convs = nn.ModuleList([nn.Conv2D(num_channel_orignal[r], num_channel_orignal[r] + num_channel_orignal[r+1] 
			, self.kernel_size, padding=self.kernel_size//2) for r in range(1, self.num_rows-1)])

		self.top_conv = nn.Conv2D(2*self.L+8, num_channel_orignal[0] + num_channel_orignal[1],
		 , self.kernel_size, padding=self.kernel_size//2)


		# Last conv
		self.last_conv1 = nn.Conv2D(self.L, 2*self.L+8,
		 , self.kernel_size, padding=self.kernel_size//2)

		self.last_conv2 = nn.Conv2D(self.L, self.L,
		 , self.last_kernel_size, padding=self.last_kernel_size//2)



	def forward(self, inp):

		# Scale, convolve and output rows of features maps
		# After this: inps = feature maps of [row1, row2, row3, row4]
		inps = [self.inp_convs[0](inp)]
		for i in range(1, self.num_rows):
			inps.append(self.inp_convs[i](F.max_pool2d(inp, kernel_size=self.kernel_size, stride=2**i, padding=self.kernel_size//2)))


		# Step forward for each interlinking layer
		for i in range(self.num_interlink_layer):

			# For each row
			interlink
			for j in range(num_rows):
				# Interlink

				# Convolve

		inps =

		# Output integration
		for x in range(num_rows):
			pass

		inp = 

		# Convolve

		# final

		return

