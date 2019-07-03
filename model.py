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
		self.inter_convs1 = nn.ModuleList([nn.Conv2D(num_channel_orignal[r], num_channel_interlinked[r], self.kernel_size, padding=self.kernel_size//2) for r in range(self.num_rows)])
		self.inter_convs2 = nn.ModuleList([nn.Conv2D(num_channel_orignal[r], num_channel_interlinked[r], self.kernel_size, padding=self.kernel_size//2) for r in range(self.num_rows)])
		self.inter_convs2 = nn.ModuleList([nn.Conv2D(num_channel_orignal[r], num_channel_interlinked[r], self.kernel_size, padding=self.kernel_size//2) for r in range(self.num_rows)])


		# Output convs
		self.out_convs = nn.ModuleList([nn.Conv2D(num_channel_orignal[r], num_channel_orignal[r] + num_channel_orignal[r+1] 
			, self.kernel_size, padding=self.kernel_size//2) for r in range(1, self.num_rows-1)])

		self.top_conv = nn.Conv2D(2*self.L+8, num_channel_orignal[0] + num_channel_orignal[1],
		 , self.kernel_size, padding=self.kernel_size//2)


		# Last conv
		self.last_conv = nn.Conv2D(self.L, 2*self.L+8,
		 , self.last_kernel_size, padding=self.last_kernel_size//2)



	def forward(self, inp):

		# Scale

		# Rows of input
		inps = 

		# Step forward for each interlinking layer
		for i in range(self.num_interlink_layer):

			# Do interlinking

			# Process each row 
			for j in range(num_rows):
				pass

		inps =

		# Output integration
		for x in range(num_rows):
			pass

		inp = 

		# Convolve

		# final

		return

