import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import torch.utils.data as utils
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torchvision.utils as vutils
import argparse
from PIL import Image
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str)
parser.add_argument('--dataset', type=str, choices=['3dshapes', 'dsprites', 'celeba'], required=True, help='Choose the dataset: 3dshapes, dsprites, or celeba')
args = parser.parse_args()

device = 0



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


    
def show(file_name, img, dpi=300):
    npimg = np.transpose(img.numpy(), (1,2,0))
    plt.figure(dpi=dpi)
    plt.title(file_name, fontsize=14)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(npimg)
    plt.show()
    
def show_subplot(file_name, img):
    npimg = np.transpose(img.numpy(), (1,2,0))
    plt.title(file_name, fontsize=25)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(npimg)

def show_and_save(file_name, img, dpi=300):
    npimg = np.transpose(img.numpy(), (1,2,0))
    f = "./%s.png" % file_name
    fig = plt.figure(dpi=dpi)
    plt.title(file_name, fontsize=14)
    plt.xticks([], [])
    plt.yticks([], [])
    #plt.imshow(npimg)
    plt.imsave(f, npimg)
    

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


        self.labels = dataset['labels']
   
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
    train_dataset = CelebADataset(imgs_path="/home/Downloads/stargan-v2/data/celeba_hq/train", 
                                attr_path="/home/Documents/data/df_attr_1.csv",
                                transform=None, crop=True, img_names=None)
    transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ]) # + activation Tanh in decoder
    train_dataset.transform = transform
    #print(len(train_dataset))
elif args.dataset == '3dshapes':
    transform = transforms.Compose([
    transforms.Resize((64, 64)),
    #transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
    # transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = Disentangled3dDataset(transform=transform)
elif args.dataset == 'dsprites':
    transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.0002,), (0.0008,))
                ]) 
    train_dataset = DisentangledSpritesDataset(transform=transform)




batch_size = 64

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


data, attr = next(iter(train_loader))
print(data.size())
print(attr.shape)


def plot_results(epoch):
    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.title("MSE")
    plt.plot(mse_losses, label="MSE")
    plt.legend(loc="lower right")
    plt.subplot(132)
    plt.title("KL_Z")
    plt.plot(z_kl_losses, label="KL Z")
    plt.legend(loc="lower right")
    plt.subplot(133)
    plt.title("KL_W")
    plt.plot(w_kl_losses, label="KL W")
    plt.legend(loc="lower right")
    plt.show()
    
    plt.figure(figsize=(20, 10), dpi=200)
    fixed_data, fixed_attr = fixed_batch
    
    plt.subplot(121)
    #show_subplot('Real_epoch_%d' % epoch, make_grid((fixed_data[:16].data * 0.5 + 0.5).cpu(), 4))
    vutils.save_image(fixed_data.cpu().data * 0.5 + 0.5,
        os.path.join(save_imgs_path, 'real_img.png'),
        normalize=True)
    
    # here we show recovered imgs
    plt.subplot(122)
    rec_imgs = model.reconstruct_x(fixed_data.to(device).float(), fixed_attr.to(device).float())
    vutils.save_image(rec_imgs.cpu().data * 0.5 + 0.5,
            os.path.join(save_imgs_path, 'rec_img_epoch_%d.png' % epoch),
            normalize=True)



if args.dataset == 'dsprites':
    input_shape = (1, 64, 64)
    labels_dim = 6
    z_dim = 5
    w_dim = 5
    max_epochs = 150
elif args.dataset =='3dshapes':
    input_shape = (3, 64, 64)
    labels_dim = 6
    z_dim = 10
    w_dim = 10
    max_epochs = 200
elif args.dataset =='celeba':
    input_shape = (3, 64, 64)
    labels_dim = 40
    z_dim = 64
    w_dim = 64
    max_epochs = 1000


lr = 3e-4

# hopefully this makes sense
beta1 = 10000 # data_term
beta2 = 0.001 # w_kl
beta3 = 0.1 # z_kl
beta4 = 10
beta5 = 1

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="CSVAE",

    # track hyperparameters and run metadata
    config={
    "tag": args.tag,
    "learning_rate": lr,
    "z_dim": z_dim,
    "w_dim": w_dim,
    "dataset": args.dataset,
    "epochs": max_epochs,
    "input_shape":input_shape,
    }
)

fixed_batch = next(iter(train_loader))




model = CSVAE(input_shape=input_shape, labels_dim=labels_dim, z_dim=z_dim, w_dim=w_dim).to(device)
model = model.train()

opt = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))
n_epochs = max_epochs

torch.autograd.set_detect_anomaly(True)

save_imgs_path = os.path.join("res",args.tag,'reconstruct')
os.makedirs(save_imgs_path, exist_ok=True)


mse_losses = []
z_kl_losses = []
w_kl_losses = []
for epoch_i in range(n_epochs):
    for cur_data, cur_attr in tqdm(train_loader):
        cur_data, cur_attr = cur_data.to(device).float(), cur_attr.to(device).float()
        opt.zero_grad()
        loss_val, recon_loss_val, z_kl_loss_val, w_kl_loss_val = model.calculate_loss(
            cur_data, cur_attr, average=True, beta1=beta1, beta2=beta2, beta3=beta3, beta4=beta4, beta5=beta5)
        loss_val.backward()
        opt.step()
        mse_losses.append(recon_loss_val.item())
        z_kl_losses.append(z_kl_loss_val.item())
        w_kl_losses.append(w_kl_loss_val.item())
    scheduler.step()
    
    plot_results(epoch_i)
    print('Epoch {}'.format(epoch_i))
    mean_mse = np.array(mse_losses[-len(train_loader):]).mean()
    mean_z_kl = np.array(z_kl_losses[-len(train_loader):]).mean()
    mean_w_kl = np.array(w_kl_losses[-len(train_loader):]).mean()
    print('Mean MSE: {:.4f}, scaled MSE: {:.4f}'.format(mean_mse, beta1 * mean_mse))
    print('Mean KL W: {:.4f}, scaled KL W: {:.4f}'.format(mean_w_kl, beta2 * mean_w_kl))
    print('Mean KL Z: {:.4f}, scaled KL Z: {:.4f}'.format(mean_z_kl, beta3 * mean_z_kl))
    print()

    models_dir = os.path.join('res', args.tag, 'models')
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, f'model_{epoch_i}.pt'))

    wandb.log({"train_loss": loss_val, "Mean MSE":mean_mse, "Scaled Mean MSE":beta1 * mean_mse, "Mean KL W":mean_w_kl, "scaled KL W":beta2 * mean_w_kl,"Mean KL Z":mean_z_kl, "scaled KL Z":beta3 * mean_z_kl})


wandb.finish()
