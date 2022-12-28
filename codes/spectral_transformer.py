import PIL
import time, math
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):

    def __init__(self, dim, heads, dim_heads, dropout):
        super().__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.scale = dim_heads ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dim_heads=dim_heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SpectralTransNet(nn.Module):
    def __init__(self, time_emb_dim=32, patch_size=5, spectral_size=200, dim=64, depth=5, heads=4, dim_heads=16, mlp_dim=8, dropout=0., use_conv=False):
        super(SpectralTransNet, self).__init__()
        self.spectral_size = spectral_size 
        image_size = patch_size * patch_size

        # 依据光谱连续性降低spectral维度 同时引入空间信息 200-3+2*2
        self.new_spectral_size = self.spectral_size
        self.new_image_size = image_size

        self.use_conv = use_conv
        if self.use_conv:
            conv3d_kernal_size = [3,3,3]
            conv3d_stride = [2,1,1]
            conv3d_padding = [1,1,1]
            self.conv3d_for_spectral_trans = nn.Sequential(
                nn.Conv3d(1, out_channels=1, kernel_size=conv3d_kernal_size, stride=conv3d_stride, padding=conv3d_padding),
                nn.ReLU(),
            )

            self.new_spectral_size = int((self.spectral_size - conv3d_kernal_size[0] + 2 * conv3d_padding[0]) / conv3d_stride[0]) + 1
            self.new_image_size = image_size
            print("new_spectral_size", self.new_spectral_size)
            print("new_image_size", self.new_image_size)

        self.spectral_patch_embedding = nn.Linear(self.new_image_size, dim)
        self.local_trans_spectral = Transformer(dim=dim, depth=depth, heads=heads, dim_heads=dim_heads, mlp_dim=mlp_dim, dropout=dropout)
        self.spectral_pos_embedding = nn.Parameter(torch.randn(1, self.new_spectral_size, dim)) # do not use cls token 

        mlp_head_dim = dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_head_dim),
            nn.Linear(mlp_head_dim, self.new_image_size)
        )

        self.to_latent_spectral = nn.Identity()
        self.time_mlp1 = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
        )
        self.time_mlp2 =  nn.Linear(time_emb_dim, 1)
        self.relu = nn.ReLU()


    def forward(self, x, emb):
        '''
        x: (batch, p, w, h), s=spectral, w=weigth, h=height

        '''
        time_emb = self.time_mlp1(emb)
        time_emb = self.relu(self.time_mlp2(time_emb))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 3]
        # print(time_emb.shape)

        x_spectral = x + time_emb
        b, c, s, w, h = x_spectral.shape
        img = w * h
        #0. Conv
        if self.use_conv:
            x_spectral = self.conv3d_for_spectral_trans(x_spectral)
        #1. reshape
        x_spectral = torch.squeeze(x_spectral, 1) #(batch, p, w, h)
        x_spectral = rearrange(x_spectral, 'b s w h-> b s (w h)') # (batch, s, w*h)

        #2. patch_embedding
        x_spectral = self.spectral_patch_embedding(x_spectral) #(batch, s`, dim)

        #3. local transformer
        # cls_tokens_spectral = repeat(self.cls_token_spectral, '() n d -> b n d', b = b) #[b,1,dim]
        # x_spectral = torch.cat((cls_tokens_spectral, x_spectral), dim = 1) #[b,s`,dim]
        x_spectral = x_spectral + self.spectral_pos_embedding[:,:]

        x_spectral = self.local_trans_spectral(x_spectral) #(batch, s`, dim)

        x_spectral = self.mlp_head(x_spectral)
        logit_spectral = self.to_latent_spectral(rearrange(x_spectral, 'b s (w h) -> b s w h', w=w, h=h) ) #(batch, s`, image_size)
        logit_spectral = torch.unsqueeze(logit_spectral, 1)

        # print(logit_spectral.shape)
        return logit_spectral


        
if __name__ == '__main__':
    model = SpectralTransNet()
    model.eval()
    print(model)
    input = torch.randn(3, 1, 200, 5, 5)
    y = model(input)
    print(y)
