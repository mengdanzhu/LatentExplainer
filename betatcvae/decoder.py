import numpy as np

import torch
from torch import nn


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    if model_type=='Csvae':
        return eval("Decoder{}X".format(model_type)),eval("Decoder{}Y".format(model_type))
    else:
        return eval("Decoder{}".format(model_type))


class DecoderControlvae(nn.Module):
    def __init__(self, img_size,
                 latent_dim_z=10):
    
        super(DecoderControlvae, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 512

        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # self.latent_dim=latent_dim_z+num_prop
        self.latent_dim_z = latent_dim_z
        self.sigmoid=torch.nn.Sigmoid()
            
        
        # Fully connected layers
        self.lin1 = nn.Linear(self.latent_dim_z, hidden_dim).cuda()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim).cuda()
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape)).cuda()

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs).cuda()

        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()

        
    
    def forward(self, z):
        batch_size = z.size(0)
                
        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))

        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x

 
