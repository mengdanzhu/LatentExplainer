import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.utils as vutils
import argparse
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import make_grid , save_image
from dataset import (ATTR_TO_IX_DICT, ATTR_IX_TO_KEEP, IX_TO_ATTR_DICT, N_ATTRS, 
                     FaceData_with_Attributes, tensor_to_attributes)

class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape=shape
    def forward(self,input):
        return input.view(self.shape)
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)
    
class Conv_block(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, kernel_size, stride=1, padding=0, negative_slope=0.2, p=0.04, transpose=False):
        super(Conv_block, self).__init__()
        
        self.transpose = transpose
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            
        self.activation = nn.LeakyReLU(negative_slope, inplace=True)
        self.dropout = nn.Dropout2d(p)
        self.batch_norm = nn.BatchNorm2d(num_features)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if not self.transpose:
            x = self.dropout(x)
        x = self.batch_norm(x)

        return x


class CSVAE(nn.Module):
    def __init__(self, input_shape, labels_dim, z_dim, w_dim, KOF=64, p=0.04):
        super(CSVAE, self).__init__()
        self.input_shape = input_shape
        self.labels_dim = labels_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        # x to x_features_dim
        self.encoder = nn.Sequential()
        self.encoder.add_module("block01", Conv_block(KOF, input_shape[0], KOF, 4, 2, 1, p=p))
        self.encoder.add_module("block02", Conv_block(KOF*2, KOF, KOF*2, 4, 2, 1, p=p))
        self.encoder.add_module("block03", Conv_block(KOF*4, KOF*2, KOF*4, 4, 2, 1, p=p))
        self.encoder.add_module("block04", Conv_block(KOF*8, KOF*4, KOF*8, 4, 2, 1, p=p))
        self.encoder.add_module("block05", Conv_block(KOF*16, KOF*8, KOF*16, 4, 2, 1, p=p))
        self.encoder.add_module("block06", Conv_block(KOF*16, KOF*16, KOF*16, 4, 2, 1, p=p))
#         self.encoder.add_module("block07", Conv_block(KOF*32, KOF*16, KOF*32, 4, 2, 1, p=p))
#         self.encoder.add_module("block08", Conv_block(KOF*32, KOF*32, KOF*32, 4, 2, 1, p=p))
        self.encoder.add_module("flatten", Flatten())
    
        x_features_dim = KOF * 8 * 2
        
        self.encoder_xy_to_w = nn.Sequential(
            nn.Linear(x_features_dim + labels_dim, w_dim), 
            nn.ReLU(), 
        )
        self.mu_xy_to_w = nn.Linear(w_dim, w_dim)
        self.logvar_xy_to_w = nn.Linear(w_dim, w_dim)
        
        self.encoder_x_to_z = nn.Sequential(
            nn.Linear(x_features_dim, z_dim), 
            nn.ReLU(), 
        )
        self.mu_x_to_z = nn.Linear(z_dim, z_dim)
        self.logvar_x_to_z = nn.Linear(z_dim, z_dim)
        
        self.encoder_y_to_w = nn.Sequential(
            nn.Linear(labels_dim, w_dim), 
            nn.ReLU(), 
#             nn.Linear(w_dim, w_dim), 
#             nn.ReLU()
        )
        self.mu_y_to_w = nn.Linear(w_dim, w_dim)
        self.logvar_y_to_w = nn.Linear(w_dim, w_dim)
        
        # Add sigmoid or smth for images!
        # (z+w) to x_sample
        # (!) no logvar for x
        self.decoder_zw_to_x = nn.Sequential()
        self.decoder_zw_to_x.add_module("block00", nn.Sequential(
            nn.Linear(z_dim+w_dim, z_dim+w_dim), 
            nn.BatchNorm1d(z_dim+w_dim), 
            nn.LeakyReLU(0.2)
        ))
        self.decoder_zw_to_x.add_module("reshape", Reshape((-1, z_dim+w_dim, 1, 1)))
        
        self.decoder_zw_to_x.add_module("block01", Conv_block(KOF*4, z_dim+w_dim, KOF*4, 4, 1, 0, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block02", Conv_block(KOF*4, KOF*4, KOF*4, 4, 2, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block03", Conv_block(KOF*2, KOF*4, KOF*2, 3, 1, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block04", Conv_block(KOF*2, KOF*2, KOF*2, 4, 2, 1, p=p, transpose=True))
#         self.decoder_zw_to_x.add_module("block05", Conv_block(KOF*4, KOF*4, KOF*4, 4, 2, 1, p=p, transpose=True))
#         self.decoder_zw_to_x.add_module("block06", Conv_block(KOF*2, KOF*4, KOF*2, 4, 2, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block05", Conv_block(KOF, KOF*2, KOF, 4, 2, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block06", Conv_block(KOF, KOF, KOF, 4, 2, 1, p=p, transpose=True))
#         self.decoder_zw_to_x.add_module("block07", nn.Sequential(
#                     nn.ConvTranspose2d(KOF, 3, 3, 1, 1)))

        self.mu_zw_to_x = nn.Sequential(
            nn.ConvTranspose2d(KOF, input_shape[0], 3, 1, 1),
            nn.Tanh()
        )
        self.logvar_zw_to_x = nn.Sequential(
            nn.ConvTranspose2d(KOF, input_shape[0], 3, 1, 1),
#             nn.Tanh()
        )
#         self.logvar_zw_to_x = nn.Linear(z_dim+w_dim, input_dim)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        
    def q_zw(self, x, y):
        """
        VARIATIONAL POSTERIOR
        :param x: input image
        :return: parameters of q(z|x), (MB, hid_dim)
        """

        x_features = self.encoder(x)
        
        intermediate = self.encoder_x_to_z(x_features)
        z_mu = self.mu_x_to_z(intermediate)
        z_logvar = self.mu_x_to_z(intermediate)
        
        xy = torch.cat([x_features, y], dim=1)
        
        intermediate = self.encoder_xy_to_w(xy)
        w_mu_encoder = self.mu_xy_to_w(intermediate)
        w_logvar_encoder = self.mu_xy_to_w(intermediate)
        
        intermediate = self.encoder_y_to_w(y)
        w_mu_prior = self.mu_y_to_w(intermediate)
        w_logvar_prior = self.mu_y_to_w(intermediate)
        
        return w_mu_encoder, w_logvar_encoder, w_mu_prior, \
               w_logvar_prior, z_mu, z_logvar
    
    def p_x(self, z, w):
        """
        GENERATIVE DISTRIBUTION
        :param z: latent vector          (MB, hid_dim)
        :return: parameters of p(x|z)    (MB, inp_dim)
        """
        
        zw = torch.cat([z, w], dim=1)
        
        intermediate = self.decoder_zw_to_x(zw)
        mu = self.mu_zw_to_x(intermediate)
        logvar = self.logvar_zw_to_x(intermediate)
        
        return mu, logvar

    def forward(self, x, y):
        """
        Encode the image, sample z and decode 
        :param x: input image
        :return: parameters of p(x|z_hat), z_hat, parameters of q(z|x)
        """
        w_mu_encoder, w_logvar_encoder, w_mu_prior, \
            w_logvar_prior, z_mu, z_logvar = self.q_zw(x, y)
        w_encoder = self.reparameterize(w_mu_encoder, w_logvar_encoder)
        w_prior = self.reparameterize(w_mu_prior, w_logvar_prior)
        z = self.reparameterize(z_mu, z_logvar)
        
        x_mu, x_logvar = self.p_x(z, w_encoder)
        return x_mu, x_logvar, \
               w_mu_encoder, w_logvar_encoder, w_mu_prior, \
               w_logvar_prior, z_mu, z_logvar

    def reconstruct_x(self, x, y):
        x_mu, x_logvar, w_mu_encoder, w_logvar_encoder, w_mu_prior, w_logvar_prior, z_mu, z_logvar = self.forward(x, y)
        return x_mu

    def calculate_loss(self, x, y, average=True, 
                       beta1=20, beta2=1, beta3=0.2, beta4=10, beta5=1):
        """
        Given the input batch, compute the negative ELBO 
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :return: -RE + beta * KL  (MB, ) or (1, )
        """
        x_mu, x_logvar, \
            w_mu_encoder, w_logvar_encoder, w_mu_prior, \
            w_logvar_prior, z_mu, z_logvar = self.forward(x, y)
        
        z_dist = dists.MultivariateNormal(z_mu.flatten(), 
                                          torch.diag(z_logvar.flatten().exp()))
        z_prior = dists.MultivariateNormal(torch.zeros(self.z_dim * z_mu.size()[0]).to(z_mu), 
                                           torch.eye(self.z_dim * z_mu.size()[0]).to(z_mu))
        
        w_dist = dists.MultivariateNormal(w_mu_encoder.flatten(), torch.diag(w_logvar_encoder.flatten().exp()))
        w_prior = dists.MultivariateNormal(w_mu_prior.flatten(), torch.diag(w_logvar_prior.flatten().exp()))
        
        z_kl = dists.kl.kl_divergence(z_dist, z_prior)
        w_kl = dists.kl.kl_divergence(w_dist, w_prior)

        recon = ((x_mu - x)**2).mean(dim=(1))
        # alternatively use predicted logvar too to evaluate density of input
        
        ELBO = beta1 * recon + beta3 * z_kl + beta2 * w_kl
        
        if average:
            ELBO = ELBO.mean()
            recon = recon.mean()
            z_kl = z_kl.mean()
            w_kl = w_kl.mean()

        return ELBO, recon, z_kl, w_kl


    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)


###visualization used here####
def show(name, img):
    npimg = img.numpy().transpose(1, 2, 0) 
    #plt.figure(dpi=300)
    plt.figure(figsize=(5 , 1)) 
    #plt.title(file_name, fontsize=fontsize)
    plt.axis('off')
    plt.imshow(npimg)
    plt.tight_layout() 
    plt.savefig(name)


###other visualization, not used here####
def imshow(tensor):
    tensor = tensor.clone()
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    #print("Shape before transpose:", tensor.shape)  # Debug: Print the shape
    print(tensor.shape)
    #tensor = tensor.squeeze(0)
    #print("Shape after squeeze:", tensor.shape)  # Debug: Print the shape

    # Reshape (C, H, W) -> (H, W, C)
    tensor = tensor.transpose((1, 2, 0))
    #tensor = np.clip(tensor, 0, 1) # Values may be out of range due to rounding errors in transformations, this line fixes that

    plt.imshow(tensor, cmap='gray')
    # plt.savefig(path)

def show_grid(images, save_path):

    plt.figure(figsize=(5 , 1))  # Adjust this based on your desired final image size
    for i, img in enumerate(images, 1):  # Start counting from 1
        plt.subplot(1, 5, i)
        plt.axis('off')  # Turn off axis numbers
        imshow(img)

    plt.tight_layout() 
    plt.savefig(save_path)
##############################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['3dshapes', 'dsprites', 'celeba'], required=True, help='Choose the dataset: 3dshapes, dsprites, or celeba')
args = parser.parse_args()

if args.dataset == 'celeba':
    input_shape = (3, 64, 64)
    labels_dim = 40
    z_dim = 64
    w_dim = 64
    model_path = "res/tryy/models/model_828.pt"
elif args.dataset == '3dshapes':
    input_shape = (3, 64, 64)
    labels_dim = 6
    z_dim = 10
    w_dim = 10
    model_path = "res/3dshapes_latent10_epoch100/models/model_92.pt"
else:
    input_shape = (1, 64, 64)
    labels_dim = 6
    z_dim = 10
    w_dim = 10
    model_path = "res/dsprites_latent10_epoch200/models/model_45.pt"
print(input_shape)

model = CSVAE(input_shape=input_shape, labels_dim=labels_dim, z_dim=z_dim, w_dim=w_dim)
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.eval()

class CelebADataset(Dataset):
    def __init__(self, imgs_path="../../img_align_celeba/img_align_celeba/img_align_celeba_cropped/", 
                 attr_path=None, transform=None, crop=False, img_names=None):
        self.imgs_path = imgs_path
        self.attr_path = attr_path
        self.transform = transform
        self.crop = crop
        if img_names is None:
            self.img_names = [filename for filename in os.listdir(self.imgs_path) if filename.endswith('.jpg')]
        else:
            self.img_names = [filename for filename in self.img_names if filename.endswith('.jpg')]

        
        # load dataset (whole untransformed images as np.arrays) into memory
        def get_img_array(img_path, crop=self.crop):
            img = Image.open(img_path).copy()
            if crop:
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                img = img.crop((32, 32, 224, 224)) # crop faces
                img = img.resize((64, 64), Image.Resampling.LANCZOS)
            return img
        
        self.attributes = None
        if attr_path is not None:
            self.attributes = pd.read_csv(attr_path)
            self.attributes = self.attributes.set_index("image_id")
            self.attributes = self.attributes.loc[self.img_names].values
            #print('a:',self.attributes) 
            self.attributes   = np.array(self.attributes).astype(int)
            self.attributes[self.attributes < 0] = 0
            self.attributes = torch.from_numpy(self.attributes).float()
            #print('b:',self.attributes)
        
        self.dataset = [get_img_array(os.path.join(self.imgs_path, i)) for i in tqdm(self.img_names)]

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img) # here should be ToTensor
            
        if self.attributes is not None:
            attr = torch.FloatTensor(self.attributes[idx])
            
        return (img, attr) if self.attributes is not None else img
        # should we return attributes? (NO)
        
    def __len__(self):
        return len(self.dataset)

class Disentangled3dDataset(Dataset):

    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        #self.filepath = f'{self.filename}'
        #dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')

        #dataset = h5py.File('3dshapes.h5', 'r')
        dataset = np.load('data/3dshapes.npz')
        self.imgs = dataset['images']
        #print('sample: ', self.imgs)


        self.labels = dataset['labels']
        #print('label: ', self.labels)
   

        # print('Metadata: \n', self.metadata)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx]
        sample = Image.fromarray((sample).astype('uint8'))

        if self.transform:
            sample = self.transform(sample)
        label = self.labels[idx].copy()
        return sample, label

class DisentangledSpritesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform=None):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filename = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.filepath = f'{self.filename}'
        dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')

        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)
        sample = Image.fromarray((sample).astype('uint8'))

        if self.transform:
            sample = self.transform(sample)
        label = self.latents_values[idx]
        return sample, label   

if args.dataset == 'celeba':

    train_dataset = CelebADataset(imgs_path="/home/Downloads/stargan-v2/data/celeba_hq/val", 
                                attr_path="/home/Documents/data/df_attr_1.csv",
                                transform=None, crop=True, img_names=None)
    # print(len(train_dataset))

    attr_of_interest_to_idx = {
        'Bangs': 5,
        'Bald': 4,
        'Smiling': 31, 
        'Eyeglasses': 15,
        'Young':39,
        'BlackHair':8,
        'BlondHair':9,
        'Mustache':22,
        'Mouth_Slightly_Open':21,
        'Male':20}


    batch_size = 64

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ]) # + activation Tanh in decoder

    train_dataset.transform = transform

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    #smiling
    attr_name = 'Bangs'
    ind = attr_of_interest_to_idx[attr_name]
    W_with = []
    W_no = []

    # Calculating the mean Z for samples with 'Smiling' and without 'Smiling'
    device=torch.device("cuda:0")

    for i, (image, attrs) in tqdm(enumerate(train_loader)):
        image = image.float().to(device)
        attrs = attrs.float().to(device)

        x_mu_1, x_logvar_1, w_mu_encoder_1, w_logvar_encoder_1, w_mu_prior_1, w_logvar_prior_1, z_mu_1, z_logvar_1 = model(image, attrs)
        
        w_hat = model.reparameterize(w_mu_encoder_1, w_logvar_encoder_1) # Return the sampled latent vector
        z_hat = model.reparameterize(z_mu_1, z_logvar_1) # Return the sampled latent vector


        
        for j in range(w_hat.shape[0]):
            
            if attrs[j, ind] == 1:
                W_with.append(w_hat[j])
                
            if attrs[j, ind] == 0:
                W_no.append(w_hat[j])
                
        if i == 1000:
            break

    W_with_mean = torch.mean(torch.stack(W_with), dim=0)
    W_no_mean = torch.mean(torch.stack(W_no), dim=0)

    W_move = W_with_mean - W_no_mean
    print('W_move:',W_move)
    z_hat = torch.randn(1, z_dim).cuda()
    print('z_hat:',z_hat)
    w_hat = torch.randn(1, w_dim).cuda()

    
    attrs_recons = []
    for alpha in [0,1,2,3,4]:
        w_hat_i = w_hat + alpha * W_move

        mu_x, logvar_x = model.p_x(z_hat, w_hat_i)

        attrs_recons.append(mu_x*0.5+0.5)

    torch.stack(attrs_recons, dim=1)[0].shape
    ZW = torch.stack(attrs_recons, dim=1)[0]
    ZW = ZW.detach()

    show(f"{attr_name}.png",
            make_grid(ZW.cpu(), 5,padding=4,pad_value=255))
    




