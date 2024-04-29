import torch
import torch.nn as nn
from src.denoising_diffusion_pytorch_1d import Unet1D
from src.diffusion_module2 import p_losses, sample

device = 'cuda'

class ConditionalDiffusionTrainingNetwork(nn.Module):

	def __init__(self,nr_feats, window_size, batch_size, noise_steps, denoise_steps, distill=False, train=True, device='cuda',sliding_size=1,loss_type='normal_l2'):
		super().__init__()
		self.dim = min(nr_feats, 16)
		self.nr_feats = nr_feats
		self.window_size = window_size
		self.batch_size = batch_size
		self.distill = distill
		self.training = train
		self.timesteps = noise_steps
		self.denoise_steps = denoise_steps
		self.denoise_fn = model = Unet1D(
			dim = 64,
			dim_mults = (1, 2, 4, 8),
			channels = nr_feats
		)
		self.sliding_size = sliding_size
		self.device = device
		self.loss_type = loss_type

	def forward(self, x, teacher=None):
		diffusion_loss = None
		x_recon = None
		x = x.transpose(2,1)

		if self.training:
			t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
			diffusion_loss = p_losses(self.denoise_fn, x, t, distill=self.distill, teacher=teacher, loss_type=self.loss_type, sliding_size=self.sliding_size)
		else:
			x_recon = sample(self.denoise_fn, shape=(x.shape[0], 1, self.window_size, self.nr_feats), x_start=x, denoise_steps=self.denoise_steps, device=self.device)

		return diffusion_loss, x_recon