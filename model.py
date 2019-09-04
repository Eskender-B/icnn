from utils import shift
import torch
import torch.nn.functional as F
import torch.nn as nn


class ICNN(nn.Module):

	def __init__(self, output_maps=9):
		super(ICNN,self).__init__()
		self.num_rows = 4
		self.num_interlink_layer = 3
		self.sf = 2
		self.kernel_size = 5	# has to be odd (or need to change padding below)
		self.last_kernel_size = 9
		self.L = output_maps
		self.num_channel_orignal = [8*i for i in range(1, self.num_rows+1)]		# [8, 16, 24, 32]
		self.num_channel_interlinked = shift(self.num_channel_orignal, -1, 0) + self.num_channel_orignal + shift(self.num_channel_orignal, 1, 0)

		# Initial batch norm
		self.initial_bnorm = nn.ModuleList([nn.BatchNorm2d(3) for r in range(self.num_rows)])

		# Input convs
		self.inp_convs = nn.ModuleList([nn.Conv2d(3, self.num_channel_orignal[r], self.kernel_size, padding=self.kernel_size//2) for r in range(self.num_rows)])
		self.inp_bnorm = nn.ModuleList([nn.BatchNorm2d(self.num_channel_orignal[r]) for r in range(self.num_rows)])


		# Interlinking convs
		self.inter_convs_row0 = nn.ModuleList([nn.Conv2d(self.num_channel_interlinked[0], self.num_channel_orignal[0],  self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])
		self.inter_convs_row1 = nn.ModuleList([nn.Conv2d(self.num_channel_interlinked[1], self.num_channel_orignal[1], self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])
		self.inter_convs_row2 = nn.ModuleList([nn.Conv2d(self.num_channel_interlinked[2], self.num_channel_orignal[2], self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])
		self.inter_convs_row3 = nn.ModuleList([nn.Conv2d(self.num_channel_interlinked[3], self.num_channel_orignal[3], self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_interlink_layer)])

		self.inter_bnorm_row0 = nn.ModuleList([nn.BatchNorm2d(self.num_channel_orignal[0]) for i in range(self.num_interlink_layer)])
		self.inter_bnorm_row1 = nn.ModuleList([nn.BatchNorm2d(self.num_channel_orignal[1]) for i in range(self.num_interlink_layer)])
		self.inter_bnorm_row2 = nn.ModuleList([nn.BatchNorm2d(self.num_channel_orignal[2]) for i in range(self.num_interlink_layer)])
		self.inter_bnorm_row3 = nn.ModuleList([nn.BatchNorm2d(self.num_channel_orignal[3]) for i in range(self.num_interlink_layer)])



		# Output convs
		self.out_convs = nn.ModuleList([nn.Conv2d(self.num_channel_orignal[r] + self.num_channel_orignal[r+1], self.num_channel_orignal[r]
			, self.kernel_size, padding=self.kernel_size//2) for r in range(1, self.num_rows-1)])
		self.out_bnorm = nn.ModuleList([nn.BatchNorm2d(self.num_channel_orignal[r]) for r in range(1, self.num_rows-1)])

		self.top_conv = nn.Conv2d(self.num_channel_orignal[0] + self.num_channel_orignal[1], 2*self.L+8 
			, self.kernel_size, padding=self.kernel_size//2)
		self.top_bnorm = nn.BatchNorm2d(2*self.L+8 )




		# Last conv
		self.last_conv1 = nn.Conv2d(2*self.L+8, self.L , self.last_kernel_size, padding=self.last_kernel_size//2)

		
		#self.last_conv2 = nn.Conv2d(self.L, self.L, self.last_kernel_size, padding=self.last_kernel_size//2)



	def forward(self, inp):

		batch,c,h,w = inp.shape

		# Scale, convolve and output rows of features maps
		# After this: inps = feature maps of [row1, row2, row3, row4]

		scaled_inp = inp
		inps = [torch.tanh(self.inp_bnorm[0](self.inp_convs[0](self.initial_bnorm[0](scaled_inp))))]
		for i in range(1, self.num_rows):
			scaled_inp = F.avg_pool2d(scaled_inp, kernel_size=2, stride=self.sf)
			inps.append(torch.tanh(self.inp_bnorm[i](self.inp_convs[i](self.initial_bnorm[i](scaled_inp)))))


		
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
				[F.max_pool2d(row_inps_prev[r-1], kernel_size=2, stride=self.sf) if r-1>=0 else torch.Tensor().to(inp.device), # Downsample
				row_inps_prev[r],					
				F.interpolate(row_inps_prev[r+1], scale_factor=self.sf, mode='nearest') if r+1<self.num_rows !=0 else torch.Tensor().to(inp.device)],	#Upsample
				dim=1)
				
				# Convolve
				row_inps.append(torch.tanh(eval("self.inter_bnorm_row%d"%r)[i](eval("self.inter_convs_row%d"%r)[i](tmp_inp))))

		# row_inps now holds rows of feature map after interlinking layer


		# Output integration
		# After this: lower_maps holds feature maps integrated from lower rows into the top one
		lower_maps = row_inps[self.num_rows-1]
		for r in range(self.num_rows-2, -1, -1):
			tmp_inp = torch.cat(
					[# No Downsample
					row_inps[r],					
					F.interpolate(lower_maps, scale_factor=2, mode='nearest')],	#Upsample
					dim=1)

			if r!=0:
				lower_maps = torch.tanh(self.out_bnorm[r-1](self.out_convs[r-1](tmp_inp)))
			else: # r=0
				lower_maps = torch.tanh(self.top_bnorm(self.top_conv(tmp_inp)))

		# Final
		return self.last_conv1(lower_maps)



class IRCNN(nn.Module):

	def __init__(self, output_maps=9):
		super(IRCNN,self).__init__()
		self.num_channel = 8
		self.num_rows = 4
		self.time_step = 3
		self.L = output_maps
		self.kernel_size = 5	# has to be odd (or need to change padding below)
		self.last_kernel_size = 9

		self.initial_bnorm = nn.BatchNorm2d(3)
		self.initial_conv = nn.Conv2d(3, self.num_channel,  self.kernel_size, padding=self.kernel_size//2)
		self.inp_bnorm = nn.ModuleList([nn.BatchNorm2d(self.num_channel) for r in range(self.num_rows)])
		self.forward_conv3 = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_channel,  self.kernel_size, padding=self.kernel_size//2) for i in [0]])
		self.forward_conv2 = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_channel,  self.kernel_size, padding=self.kernel_size//2) for i in [0]])
		self.forward_conv1 = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_channel,  self.kernel_size, padding=self.kernel_size//2) for i in [0]])


		self.from_up_conv2 = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_channel,  self.kernel_size, padding=self.kernel_size//2) for i in [0]])
		self.from_up_conv1 = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_channel,  self.kernel_size, padding=self.kernel_size//2) for i in [0]])
		
		self.from_down_conv3 = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_channel,  self.kernel_size, padding=self.kernel_size//2) for i in [0]])
		self.from_down_conv2 = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_channel,  self.kernel_size, padding=self.kernel_size//2) for i in [0]])
		self.from_down_conv1 = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_channel,  self.kernel_size, padding=self.kernel_size//2) for i in [0]])


		self.bnorm3 = nn.ModuleList([nn.BatchNorm2d(self.num_channel) for t in range(self.time_step)])
		self.bnorm2 = nn.ModuleList([nn.BatchNorm2d(self.num_channel) for t in range(self.time_step)])
		self.bnorm1 = nn.ModuleList([nn.BatchNorm2d(self.num_channel) for t in range(self.time_step)])



		self.down_conv = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_channel,  self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_rows-2)])
		self.straight_conv = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_channel,  self.kernel_size, padding=self.kernel_size//2) for i in range(self.num_rows-2)])
		self.out_bnorm = nn.ModuleList([nn.BatchNorm2d(self.num_channel) for i in range(self.num_rows-2)])


		# Last conv
		self.low_conv = nn.Conv2d(self.num_channel, 2*self.L+8, self.kernel_size, padding=self.kernel_size//2)
		self.low_bnorm = nn.BatchNorm2d(2*self.L+8 )

		self.last_conv1 = nn.Conv2d(2*self.L+8, self.L , self.last_kernel_size, padding=self.last_kernel_size//2)
		#self.last_conv2 = nn.Conv2d(self.L, self.L, self.last_kernel_size, padding=self.last_kernel_size//2)



	def forward(self, inp):
		inp = self.initial_conv(self.initial_bnorm(inp))
		inps = [torch.tanh(self.inp_bnorm[0](inp))]
		for r in range(1, self.num_rows):
			inps.append(torch.tanh(self.inp_bnorm[r](eval("self.forward_conv%d"%r)[0](inps[-1]))))


		for t in range(self.time_step):
			prev_inps = inps
			inps = [inps[0]]
			
			for r in range(1, self.num_rows):
				inp = eval("self.forward_conv%d"%r)[0](prev_inps[r]) + eval("self.from_down_conv%d"%r)[0](prev_inps[r-1])

				if r <= self.num_rows-2:
					inp +=  eval("self.from_up_conv%d"%r)[0](prev_inps[r+1])

				inps.append(torch.tanh(eval("self.bnorm%d"%r)[t](inp)))


		upper_maps = inps[self.num_rows-1]
		for r in range(self.num_rows-2, 0, -1):
			upper_maps = torch.tanh(self.out_bnorm[r-1](self.straight_conv[r-1](inps[r]) + self.down_conv[r-1](upper_maps)))

		tmp=torch.tanh(self.low_bnorm(self.low_conv(upper_maps)))

		return self.last_conv1(tmp)

