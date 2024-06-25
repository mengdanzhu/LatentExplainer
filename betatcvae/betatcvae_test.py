import torch
from torch.utils.data import DataLoader, Dataset,random_split
from utils.math import (log_density_gaussian, log_importance_weight_matrix,
                               matrix_log_density_gaussian)
from model_VAE import ControlVAE
from torch import optim
import time
import torch.nn.functional as F
from encoder import *
from decoder import *
from torchvision import transforms
import h5py
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
from torchvision.datasets import CelebA
from torchvision.datasets import ImageFolder
from PIL import Image


import wandb

epochs = 200
epoch = 199
rec_loss_weight = 1
kl_loss_weight = 1
pairwise_tc_loss_weight = 2
dataset_name = 'celeba' #'3dshapes','dsprites','celeba'
assert dataset_name in ('3dshapes','dsprites','celeba')
model_name = 'betatcvae' 

if dataset_name == 'dsprites':
    latent_dim = 5
elif dataset_name == '3dshapes':
    latent_dim = 6
else:
    latent_dim = 64

lr = 3e-4
batch_size = 64
num_worker = 4
save_epoch_interval = 5

torch.cuda.set_device(3)

dir_name = f'{model_name}_{dataset_name}_latent_dim_{latent_dim}_tc_loss_weight_{pairwise_tc_loss_weight}_epoch_{epochs}'
print(dir_name)



class DisentangledSpritesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform=None):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.filepath = f'{self.filename}'
        dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')

        # print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        #self.metadata = dataset_zip['metadata'][()]

        # print('Metadata: \n', self.metadata)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)
        # sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        sample = Image.fromarray((sample).astype('uint8'))
        if self.transform:
            sample = self.transform(sample)
        #torch_tensor = torch.from_numpy(sample).float()
        label = self.latents_classes[idx]

        return sample, label

class Disentangled3dDataset(Dataset):

    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        dataset = np.load('data/3dshapes.npz')
        self.imgs = dataset['images']

        self.labels = dataset['labels']

        # print('Metadata: \n', self.metadata)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx]
        sample = Image.fromarray((sample).astype('uint8'))

        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        #torch_tensor = torch.from_numpy(sample).float()
        return sample, label
    
def _kl_normal_loss(mean, logvar):
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()
    return total_kl

def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

def calc_all_loss(data, reconstruct, latent_dist_z, latent_sample_z):
    # Loss
    latent_dist=(torch.cat([latent_dist_z[0]],dim=-1),torch.cat([latent_dist_z[1]],dim=-1))

    ###### Reconstruction loss ######

    rec_loss = F.binary_cross_entropy(reconstruct, data, reduction="sum") / 200

    ###### KL loss ######
    kl_loss = _kl_normal_loss(*latent_dist)

    log_pw, log_qw, log_prod_qwi, log_q_wCx = _get_log_pz_qz_prodzi_qzCx(latent_sample_z,
                                                                    latent_dist_z,
                                                                    len(train_loader.dataset),
                                                                    is_mss=True)

    tc_loss = (log_qw - log_prod_qwi).mean()
    pairwise_tc_loss = tc_loss

    return rec_loss, pairwise_tc_loss, kl_loss

def evaluate(model, test_loader):
    epoch_loss = 0
    epoch_rec_loss = 0
    epoch_kl_loss = 0
    epoch_p_tc_loss = 0

    w_kl = 0
    for _, (data,label) in enumerate(test_loader):

        w_kl += 1
        data = data.cuda()
        reconstruct, latent_dist_z, latent_sample_z  = model(data)

        rec_loss, pairwise_tc_loss, \
            kl_loss = calc_all_loss(data, reconstruct, latent_dist_z,\
                                 latent_sample_z)

        loss = rec_loss_weight * rec_loss + pairwise_tc_loss_weight * pairwise_tc_loss + kl_loss_weight * kl_loss

        epoch_loss += loss
        epoch_rec_loss += rec_loss
        epoch_kl_loss += kl_loss
        epoch_p_tc_loss += pairwise_tc_loss
        return epoch_loss, epoch_rec_loss, epoch_kl_loss, epoch_p_tc_loss

def compute_mean_std(train_loader,dataset_name):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    for data, _ in train_loader:
        b, c, h, w = data.shape
        nb_pixels = b*h*w
        fst_moment = (cnt * fst_moment + torch.sum(data, dim=[0,2,3])) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + torch.sum(data**2, dim=[0,2,3])) / (cnt + nb_pixels)
        cnt += nb_pixels
    print(dataset_name)
    print('mean:', fst_moment)
    print('std:',torch.sqrt(snd_moment - fst_moment ** 2))


if dataset_name in ('dsprites'):
    img_size = [1,64,64]
else:
    img_size = [3,64,64]




transform = transforms.Compose([
    transforms.Resize((64, 64)),
    #transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    # transforms.Normalize((0.5,), (0.5,)),
])


if dataset_name == '3dshapes':
    dataset = Disentangled3dDataset(transform=transform)
    dataset_size = len(dataset)
    train_size = int(np.floor(0.8 * dataset_size))
    test_size = dataset_size - train_size 
    torch.manual_seed(42)
    data_train, data_test = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    #compute_mean_std(train_loader,dataset_name)

elif dataset_name == 'dsprites':  
    transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.0002,), (0.0008,))]) 
    dataset = DisentangledSpritesDataset(transform=transform)
    dataset_size = len(dataset)
    train_size = int(np.floor(0.7 * dataset_size))
    test_size = dataset_size - train_size 
    torch.manual_seed(42)
    data_train, data_test = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    #compute_mean_std(train_loader,dataset_name)

elif dataset_name == 'celeba':

    class CustomImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.jpg', '.jpeg', '.png'))]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, []

    # Load datasets
    train_dataset = CustomImageDataset(root_dir='/home/data/celeba_hq/train/', transform=transform)
    val_dataset = CustomImageDataset(root_dir='/home/data/celeba_hq/val/', transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

encoder = eval("EncoderControlvae")
decoder = eval("DecoderControlvae")

framework = ControlVAE(img_size, encoder, decoder, latent_dim).cuda()


checkpoint = torch.load("ckpt/{}/model{}{}_{}_epoch{}_{}_{}_{}.pt".format(dir_name,model_name, latent_dim, dataset_name, epoch, rec_loss_weight, kl_loss_weight, pairwise_tc_loss_weight))
framework.load_state_dict(checkpoint['model_state_dict'])

model = framework


from matplotlib import pyplot as plt

id = 0

def imshow(tensor, path):
    tensor = tensor.clone()
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    # Reshape (C, H, W) -> (H, W, C)
    tensor = tensor.transpose((1, 2, 0))
    #tensor = np.clip(tensor, 0, 1) # Values may be out of range due to rounding errors in transformations, this line fixes that

    plt.imshow(tensor)
    plt.savefig(path)

for _, (data,label) in enumerate(test_loader):

    batch_size, channel, height, width = data.size()
    data = data.cuda()
    id += 1

    reconstruct, latent_dist_z, latent_sample_z = model(data)

    dir_name_r = f'{model_name}_{dataset_name}_latent_dim_{latent_dim}_tc_loss_weight_{pairwise_tc_loss_weight}_epoch_{epoch}'


    if not os.path.exists(os.path.join('vis_reconstruct',dir_name_r)):
        os.makedirs(os.path.join('vis_reconstruct',dir_name_r))


    imshow(data[0], f"vis_reconstruct/{dir_name_r}/"+model_name+str(dataset_name)+str(id)+"_ori.png")
    imshow(reconstruct[0], f"vis_reconstruct/{dir_name_r}/"+model_name+str(dataset_name)+str(id)+"_recon.png")

    if id == 6:
        break


def imshow(tensor):
    tensor = tensor.clone()
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    # Reshape (C, H, W) -> (H, W, C)
    tensor = tensor.transpose((1, 2, 0))
    tensor = np.clip(tensor, 0, 1) # Values may be out of range due to rounding errors in transformations, this line fixes that

    plt.imshow(tensor, cmap='gray')
    # plt.savefig(path)

def show_grid(images, save_path):

    plt.figure(figsize=(5 , latent_dim))  # Adjust this based on your desired final image size
    for i, img in enumerate(images, 1):  # Start counting from 1
        plt.subplot(latent_dim, 5, i)
        plt.axis('off')  # Turn off axis numbers
        imshow(img)

    plt.tight_layout() 
    plt.savefig(save_path)

#set seed
seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

z = torch.randn(1, latent_dim).cuda()
values = torch.range(-3, 3, 1.5)
all_imgs = []

for i in range(latent_dim):
    for j in values:
        z_prime = torch.clone(z)
        z_prime[0][i] = j
        reconstruct = model.specified_sample(z_prime)
        all_imgs.append(reconstruct[0])



show_grid(all_imgs, 'vis_transverse/vis_transverse_{}{}_{}_epoch{}_{}_{}_{}.png'.format(model_name,latent_dim, dataset_name, epoch,rec_loss_weight, kl_loss_weight, pairwise_tc_loss_weight))

if dataset_name == '3dshapes':
    comparison_all_imgs = []

    beta = 0
    if (beta != z[0, 1]) and (beta != z[0, 3]):  
        for i in range(latent_dim):
            for j in values:
                z_prime = torch.clone(z)
                z_prime[0][i] = j
                if i != 1:
                    z_prime[0][1] = beta
                else:
                    z_prime[0][3] = beta

                comparison_reconstruct = model.specified_sample(z_prime)
                comparison_all_imgs.append(comparison_reconstruct[0])
       

    show_grid(comparison_all_imgs, 'vis_transverse/vis_transverse_{}{}_{}_epoch{}_{}_{}_{}_comparison.png'.format(model_name,latent_dim, dataset_name, epoch,rec_loss_weight, kl_loss_weight, pairwise_tc_loss_weight))

elif dataset_name == 'celeba':
    comparison_all_imgs = []

    beta = 3
    if (beta != z[0, 0]) and (beta != z[0, 1]):  
        for i in range(latent_dim):
            for j in values:
                z_prime = torch.clone(z)
                z_prime[0][i] = j
                if i != 0:
                    z_prime[0][0] = beta
                else:
                    z_prime[0][1] = beta

                comparison_reconstruct = model.specified_sample(z_prime)
                comparison_all_imgs.append(comparison_reconstruct[0])
        

    show_grid(comparison_all_imgs, 'vis_transverse/vis_transverse_{}{}_{}_epoch{}_{}_{}_{}_comparison.png'.format(model_name,latent_dim, dataset_name, epoch,rec_loss_weight, kl_loss_weight, pairwise_tc_loss_weight))

else:
    comparison_all_imgs = []

    beta = 0
    if (beta != z[0, 2]) and (beta != z[0, 1]):  
        for i in range(latent_dim):
            for j in values:
                z_prime = torch.clone(z)
                z_prime[0][i] = j
                if i != 2:
                    z_prime[0][2] = beta
                else:
                    z_prime[0][1] = beta

                comparison_reconstruct = model.specified_sample(z_prime)
                comparison_all_imgs.append(comparison_reconstruct[0])
        

    show_grid(comparison_all_imgs, 'vis_transverse/vis_transverse_{}{}_{}_epoch{}_{}_{}_{}_comparison.png'.format(model_name,latent_dim, dataset_name, epoch,rec_loss_weight, kl_loss_weight, pairwise_tc_loss_weight))
