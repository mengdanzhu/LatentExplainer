import torch
from torch import nn, optim
from torch.nn import functional as F


class ControlVAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim):
     
        super(ControlVAE, self).__init__()

        # if list(img_size[1:]) not in [[32, 32], [64, 64]]:
        #     raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))
        
        # Number of properties
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim)

            
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps


    def forward(self, x):

        latent_dist_z_mean,latent_dist_z_std = self.encoder(x) #for training process
        
        latent_sample_z = self.reparameterize(latent_dist_z_mean,latent_dist_z_std)

        reconstruct = self.decoder(latent_sample_z)
        
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        return reconstruct, latent_dist_z, latent_sample_z

    def sample_latent(self, x,p=None):
        
        latent_dist_z_mean,latent_dist_z_std = self.encoder(x,p)
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_sample_z = self.reparameterize(*latent_dist_z)
        
        return latent_sample_z
    
    def sample(self, num_samples):
        # z = 2 * torch.rand(num_samples, self.latent_dim).cuda() - 1
        z = torch.randn(num_samples, self.latent_dim).cuda()

        samples = self.decoder(z)
        return samples

    def specified_sample(self, z):

        samples = self.decoder(z)
        return samples
