# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import numpy as np
import PIL.Image
import torch
from torchvision.utils import make_grid

from legacy import load_network

#----------------------------------------------------------------------------

def make_dataset(dir, extension='npz'):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith(extension):
                path = os.path.join(root, fname)
                images.append(path)
    return images

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

def generate_blended_img(G_t, z=None, blend_weights=[0,0.25,0.5,0.75,1], truncation_psi=0.7, truncation_cutoff=8):
    all_images = []
    
    if z == None:
        z = torch.randn([1,512]).cuda()
    assert z.shape == torch.Size([1, 512])
    
    c = torch.zeros(1,0).cuda()

    for weight in blend_weights:
        img = G_t(z, c, truncation_psi, truncation_cutoff, noise_mode='interpolate', blend_weight=weight)
        all_images.append(img)

    all_images = torch.cat(all_images)
    images = make_grid(all_images, nrow=len(blend_weights), padding=5, pad_value=0.99999)
    images = (images.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    images = PIL.Image.fromarray(images, 'RGB')
    return images

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--cfg', help='Network configuration', default='paper256', required=True)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc_psi', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc_cutoff', 'truncation_cutoff', type=int, help='Truncation psi', default=8, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--interp-step', help='Number of interpolation steps', type=int, default=5, show_default=True)
@click.option('--projected-z-dir', help='Projection result file directory', type=str, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    cfg: str,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    truncation_cutoff: int,
    interp_step: int,
    outdir: str,
    class_idx: Optional[int],
    projected_z_dir: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate MetFaces images without truncation
    python generate.py --cfg=paper256 --outdir=out --trunc_psi=1 --seeds=0-35 \\
        --network=pretrained/metfaces-fm0.05-001612.pkl

     \b
    # Generate MetFaces images with truncation
    python generate.py --cfg=paper256 --outdir=out --trunc_psi=0.7 --trunc_cutoff=8 --seeds=0-35 \\
        --network=pretrained/metfaces-fm0.05-001612.pkl

    \b
    # Generate AAHQ images with truncation
    python generate.py --cfg=paper256 --outdir=out --trunc_psi=0.7 --trunc_cutoff=8 --seeds=0-35 \\
        --network=pretrained/aahq-fm0.05-010886.pkl

    \b
    # Generate Wikiart images with truncation
    python generate.py --cfg=stylegan2 --outdir=out --trunc_psi=0.7 --trunc_cutoff=8 --seeds=0-35 \\
        --network=pretrained/wikiart-fm0.05-004032.pkl

    \b
    # Render an image from projected Z
    python generate.py --cfg=paper256 --outdir=out --trunc_psi=0.7 --trunc_cutoff=8 \\
        --projected-z-dir=./projected --network=pretrained/aahq-fm0.05-010886.pkl
    """

    device = torch.device('cuda')
    G = load_network(cfg, network_pkl, img_resolution=256, img_channels=3, c_dim=0).to(device)

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_z_dir is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-z-dir')
        print(f'Generating images from projected Z"{projected_z_dir}"')
        z_paths = sorted(make_dataset(projected_z_dir))
        for p in z_paths:
            z = np.load(p)['z']
            z = torch.tensor(z, device=device) # pylint: disable=not-callable
            images = generate_blended_img(G, z=z, blend_weights=np.linspace(0,1,interp_step), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            basename = os.path.basename(p).split('.')[0]
            images.save(f'{outdir}/{basename}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        images = generate_blended_img(G, z=z, blend_weights=np.linspace(0,1,interp_step), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        images.save(f'{outdir}/seed{seed:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
