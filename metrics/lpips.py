import copy
import numpy as np
import torch
import dnnlib
import legacy
from . import metric_utils
from torch_utils import misc
import lpips

resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
        'church256':   'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-church-config-f.pkl',
}


class LPIPSSampler(torch.nn.Module):
    def __init__(self, G, G_orig, G_kwargs, epsilon, crop):
        super().__init__()
        self.G = copy.deepcopy(G)
        self.G_orig = copy.deepcopy(G_orig)
        self.G_kwargs = G_kwargs
        self.epsilon = epsilon
        self.crop = crop
        self.vgg16 = lpips.LPIPS(net='vgg')

    def forward(self, c):
        # Generate images.
        z = torch.rand([c.shape[0], self.G.z_dim], device=c.device)
        img = self.G(z=z, c=c, **self.G_kwargs)
        img_orig = self.G_orig(z=z, c=c, **self.G_kwargs)

        # concatenate images to batch to accelarate
        img = torch.cat([img, img_orig])

        # Center crop.
        if self.crop:
            assert img.shape[2] == img.shape[3]
            c = img.shape[2] // 8
            img = img[:, :, c*3 : c*7, c*2 : c*6]
        
        # Downsample to 256x256.
        factor = self.G.img_resolution // 256
        if factor > 1:
            img = img.reshape([-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]).mean([3, 5])

        if self.G.img_channels == 1:
            img = img.repeat([1, 3, 1, 1])
        
        # Evaluate differential LPIPS.
        img, img_orig = img.chunk(2)
        dist = self.vgg16(img, img_orig)
        return dist


def compute_lpips(opts, num_samples, epsilon, crop, batch_size, resume='ffhq', jit=False):
    assert opts.dataset_kwargs.resolution in [256, 512, 1024]

    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
   
    G_orig_url = resume_specs[resume+str(opts.dataset_kwargs.resolution)]
    G_orig = copy.deepcopy(opts.G)
    with dnnlib.util.open_url(G_orig_url) as f:
        G_orig_data = legacy.load_network_pkl(f)
    misc.copy_params_and_buffers(G_orig_data['G_ema'], G_orig, require_all=False)

    # Setup sampler.
    sampler = LPIPSSampler(G=opts.G, G_orig=G_orig, G_kwargs=opts.G_kwargs, epsilon=epsilon, crop=crop)
    sampler.eval().requires_grad_(False).to(opts.device)
    if jit:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        sampler = torch.jit.trace(sampler, [c], check_trace=False)

    # Sampling loop.
    dist = []
    progress = opts.progress.sub(tag='lpips sampling', num_items=num_samples)
    for batch_start in range(0, num_samples, batch_size * opts.num_gpus):
        progress.update(batch_start)
        c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
        c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
        x = sampler(c)
        for src in range(opts.num_gpus):
            y = x.clone()
            if opts.num_gpus > 1:
                torch.distributed.broadcast(y, src=src)
            dist.append(y)
    progress.update(num_samples)

    # Compute LPIPS.
    if opts.rank != 0:
        return float('nan')
    dist = torch.cat(dist)[:num_samples].cpu().numpy()
    lpips = dist.mean()
    return float(lpips)