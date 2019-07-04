from utils import shift
import torch
import torch.nn.functional as F
import torch.nn as nn


class ICNN(nn.Module):

	def __init__(self):
		super(ICNN,self).__init__()
		self.num_rows = 4
		self.num_interlink_layer = 3
		self.sf = 2
		self.kernel_size = 5
		self.last_kernel_size = 9
		self.L = 10
		self.num_channel_orignal = [8*i for i in range(1, self.num_rows+1)]		# [8, 16, 24, 32]
		self.num_channel_interlinked = shift(self.num_channel_orignal, -1, 0) + self.num_channel_orignal + shift(self.num_channel_orignal, 1, 0) 

		# Input convs
		self.inp_convs = nn.ModuleList([nn.Conv2d(3, self.num_channel_orignal[r], self.kernel_size, padding=self.kernel_size//2) for r in range(self.num_rows)])


		# Interlinking convs
		self.inter_convs_row0 = nn.ModuleList([nn.Conv2d(self.num_channel_interlinked[0], self.num_channel_orignal[0],  self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])
		self.inter_convs_row1 = nn.ModuleList([nn.Conv2d(self.num_channel_interlinked[1], self.num_channel_orignal[1], self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])
		self.inter_convs_row2 = nn.ModuleList([nn.Conv2d(self.num_channel_interlinked[2], self.num_channel_orignal[2], self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])
		self.inter_convs_row3 = nn.ModuleList([nn.Conv2d(self.num_channel_interlinked[3], self.num_channel_orignal[3], self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])


		# Output convs
		self.out_convs = nn.ModuleList([nn.Conv2d(self.num_channel_orignal[r] + self.num_channel_orignal[r+1], self.num_channel_orignal[r]
			, self.kernel_size, padding=self.kernel_size//2) for r in range(1, self.num_rows-1)])

		self.top_conv = nn.Conv2d(self.num_channel_orignal[0] + self.num_channel_orignal[1], 2*self.L+8 
			, self.kernel_size, padding=self.kernel_size//2)


		# Last conv
		self.last_conv1 = nn.Conv2d(2*self.L+8, self.L 
			, self.kernel_size, padding=self.kernel_size//2)

		self.last_conv2 = nn.Conv2d(self.L, self.L
			, self.last_kernel_size, padding=self.last_kernel_size//2)



	def forward(self, inp):

		batch = inp.shape[0]
		h = inp.shape[2]
		w = inp.shape[3]

		# Scale, convolve and output rows of features maps
		# After this: inps = feature maps of [row1, row2, row3, row4]

		scaled_inp = inp
		inps = [torch.tanh(self.inp_convs[0](scaled_inp))]
		for i in range(1, self.num_rows):
			scaled_inp = F.max_pool2d(scaled_inp, kernel_size=3, stride=self.sf, padding=1)
			inps.append(torch.tanh(self.inp_convs[i](scaled_inp)))


		
		# Step forward for each interlinking layer
		# After this: row_inps = feature maps of [row1, row2, row3, row4] after interlinking layer
		row_inps = inps
		for i in range(self.num_interlink_layer):

			row_inps_prev = row_inps
			row_inps = []
			
			# For each row
			for r in range(self.num_rows):

				# Interlink
				tmp_inp = torch.cat(
				[F.max_pool2d(row_inps_prev[r-1], kernel_size=3, stride=self.sf, padding=1) if r-1>=0 else torch.Tensor(device=inp.device), # Downsample
				row_inps_prev[r],					
				F.interpolate(row_inps_prev[r+1], scale_factor=self.sf, mode='nearest') if r+1<self.num_rows !=0 else torch.Tensor(device=inp.device)],	#Upsample
				dim=1)
				
				# Convolve
				row_inps.append(torch.tanh(eval("self.inter_convs_row%d"%r)[i](tmp_inp)))

		# row_inps now holds rows of feature map after interlinking layer


		# Output integration
		# After this: lower_maps holds feature maps integrated from lower rows into the top one
		out_maps = None
		lower_maps = row_inps[self.num_rows-1]
		for r in range(self.num_rows-2, -1, -1):
			tmp_inp = torch.cat(
					[# No Downsample
					row_inps[r],					
					F.interpolate(lower_maps, scale_factor=2, mode='nearest')],	#Upsample
					dim=1)

			if r!=0:
				lower_maps = torch.tanh(self.out_convs[r-1](tmp_inp))
			else: # r=0
				lower_maps = torch.tanh(self.top_conv(tmp_inp))

		# Final
		return self.last_conv2(self.last_conv1(lower_maps))