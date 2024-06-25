import numpy as np

import torch
from torch import nn

# ALL encoders should be called Encoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))
       
class EncoderControlvae(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10,if_given_property=False):
        super(EncoderControlvae, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256

        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        # 3 convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)

        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs).cuda()
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()
       

        
        # Fully connected layers for unobversed properties
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim).cuda()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim).cuda()
        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, (self.latent_dim) * 2).cuda()           

    def forward(self, x, label=None, prop=None):
        
        batch_size = x.size(0)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))     
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))        
   

        x_z = x.view((batch_size, -1))

        x_z = torch.relu(self.lin1(x_z))
        x_z = torch.relu(self.lin2(x_z))        
        
        
        mu_logvar = self.mu_logvar_gen(x_z)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar

     
