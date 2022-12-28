import torch
import torchvision
from torchvision import transforms 
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt

from new_data import HSIDataLoader, TestDS, TrainDS
from unet3d import SimpleUnet
from spectral_transformer import SpectralTransNet
from diffusion import Diffusion
from utils import AvgrageMeter, recorder, show_img

batch_size = 256
channel = 1
spe = 200
patch_size = 16
epochs = 100000 # Try more!
lr = 0.01
T=500


device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_by_imgs(imgs, rgb=[1,100,199]):
    assert len(imgs) > 0
    batch, c, s, h, w = imgs[0].shape
    for i in range(batch):
        plt.figure(figsize=(12,8))
        for j in range(len(imgs)):
            plt.subplot(1,len(imgs),j+1)
            img = imgs[j][i,0,rgb,:,:]
            show_img(img)
        plt.show()            
    
def plot_by_images_v2(imgs, rgb=[1,100,199]):
    '''
    input image shape is (spectral, height, width)
    '''
    assert len(imgs) > 0
    s,h,w = imgs[0].shape
    plt.figure(figsize=(12,8))
    for j in range(len(imgs)):
        plt.subplot(1,len(imgs),j+1)
        img = imgs[j][rgb,:,:]
        show_img(img)
    plt.show()            
    

def recon_all_fig(diffusion, model, splitX, dataloader, big_img_size=[145, 145]):
    '''
    X shape is (spectral, h, w) => (batch, channel=1, 200, 145, 145)
    '''
    # 1. reconstruct
    t = torch.full((1,), diffusion.T-1, device=device, dtype=torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(torch.from_numpy(splitX.astype('float32')), t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num = 5)
    res_xt_list = []
    for tempxt in recon_from_xt:
        big_xt = dataloader.split_to_big_image(tempxt.numpy()) 
        res_xt_list.append(big_xt)
    ori_data, _ = dataloader.get_ori_data()
    res_xt_list.append(ori_data)
    plot_by_images_v2(res_xt_list)
    

def sample_eval(diffusion, model, X):
    all_size, channel, spe, h, w = X.shape
    num = 5
    step = all_size // num
    r,g,b = 1, 100, 199
    choose_index = list(range(0, all_size, step))
    x0 = torch.from_numpy(X[choose_index,:,:,:,:]).float()

    # from xt
    t = torch.full((1,), diffusion.T-1, device=device, dtype=torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num = 10)
    recon_from_xt.append(x0)
    plot_by_imgs(recon_from_xt)
    
    # from noise
    t = torch.full((1,), diffusion.T-1, device=device, dtype=torch.long)
    _, recon_from_noise = diffusion.reconstruct(model, xt=x0, tempT=t, num = 10, from_noise=True, shape=x0.shape)
    plot_by_imgs(recon_from_noise)



def train():
    dataloader = HSIDataLoader({"data":{"padding":False, "batch_size":batch_size, "patch_size":patch_size}})
    train_loader,X,Y = dataloader.generate_torch_dataset()
    diffusion = Diffusion(T=T)
    # model = SimpleUnet(_image_channels=channel)
    model = SpectralTransNet(patch_size=patch_size, spectral_size=spe, heads=4, dim=32, depth=5, dim_heads=16, mlp_dim=8)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    loss_metric = AvgrageMeter()
    for epoch in range(epochs):
        loss_metric.reset()
        for step, (batch, _) in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            cur_batch_size = batch.shape[0]
            t = torch.randint(0, diffusion.T , (cur_batch_size,), device=device).long()
            loss, temp_xt, temp_noise, temp_noise_pred = diffusion.get_loss(model, batch, t)
            loss.backward()
            optimizer.step()
            loss_metric.update(loss.item(), batch.shape[0])

            if step % 10 == 0:
                print(f"[Epoch-step] {epoch} | step {step:03d} Loss: {loss.item()} ")
        print("[TRAIN EPOCH %s] loss=%s" % (epoch, loss_metric.get_avg()))

        if epoch % 5 == 0:
            # sample_eval(diffusion, model, X)
            _, splitX, splitY = dataloader.generate_torch_dataset(split=True)
            recon_all_fig(diffusion, model, splitX, dataloader, big_img_size=[145, 145])


if __name__ == "__main__":
    train()