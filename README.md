# Fix the Noise: Disentangling Source Feature for Controllable Domain Translation</sub>

[![arXiv](https://img.shields.io/badge/arXiv-2303.11545-b31b1b.svg)](https://arxiv.org/abs/2303.11545)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeeDongYeun/FixNoise/blob/main/demo_colab.ipynb)

![Teaser image](./docs/figure_1.png)
**Fix the Noise: Disentangling Source Feature for Controllable Domain Translation**<br>
Dongyeun Lee, Jae Young Lee, Doyeon Kim, Jaehyun Choi, Jaejun Yoo, Junmo Kim<br>
https://arxiv.org/abs/2303.11545

>**Abstract**: 
*Recent studies show strong generative performance in domain translation especially by using transfer learning techniques on the unconditional generator. However, the control between different domain features using a single model is still challenging. Existing methods often require additional models, which is computationally demanding and leads to unsatisfactory visual quality. In addition, they have restricted control steps, which prevents a smooth transition. In this paper, we propose a new approach for high-quality domain translation with better controllability. The key idea is to preserve source features within a disentangled subspace of a target feature space. This allows our method to smoothly control the degree to which it preserves source features while generating images from an entirely new domain using only a single model. Our extensive experiments show that the proposed method can produce more consistent and realistic images than previous works and maintain precise controllability over different levels of transformation.*

## Recent Updates
* **`2023-05-09`** Add several useful code for inference. For detailed usage, refer to [Inference](#Inference).

* **`2023-05-15`** Add [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeeDongYeun/FixNoise/blob/main/demo_colab.ipynb)

## Requirements
Our code is highly based on the official implementation of [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). Please refer to [requirements](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements) for detailed requirements.
* Python libraries:
```bash
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 lpips
```
* Docker users:
```bash
docker build --tag sg2ada:latest .
docker run --gpus all --shm-size 64g -it -v /etc/localtime:/etc/localtime:ro -v /mnt:/mnt -v /data:/data --name sg2ada sg2ada /bin/bash
```


## Pretrained Checkpoints
You can download the pre-trained checkpoints used in our paper:
| Setting                 |   Resolution  | Config   |    Description   |
| :--------------------   | :------------ | :------- | :--------------- |
| [FFHQ &rarr; MetFaces](https://drive.google.com/file/d/1Eo4T9KjkzRYdnENXgTpqIUOvaY4-SDeD/view?usp=sharing)    |    256x256    | paper256 | Trained initialized with official [pre-trained model on FFHQ 256](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/) from Pytorch implementation of [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). |
| [FFHQ &rarr; AAHQ](https://drive.google.com/file/d/1GzM3icWaSOSGcKfYoidjEaloqc_MyAxX/view?usp=sharing)        |    256x256    | paper256 | Trained initialized with official [pre-trained model on FFHQ 256](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/) from Pytorch implementation of [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). |
| [Church &rarr; Cityscape](https://drive.google.com/file/d/1YHa_g5xC_VM5MbHsr3VSfco1_PX1sRkA/view?usp=sharing) |    256x256    | stylegan2| Trained initialized with official [pre-trained model on LSUN Church config-f](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/) from Tensorflow implementation of [stylegan2](https://github.com/NVlabs/stylegan2). |

## Datasets
We provide official dataset download pages and our processing code for reproducibility. You could alse use official processing code in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch#preparing-datasets). However, doing so does not guarantee reported performance.

**MetFaces**: Download the [MetFaces dataset](https://github.com/NVlabs/metfaces-dataset) and unzip it.
```bash
# Resize MetFaces
python dataset_resize.py --source data/metfaces/images --dest data/metfaces/images256x256
```
**AAHQ**: Download the [AAHQ dataset](https://github.com/onion-liu/aahq-dataset) and process it following original instruction.
```bash
# Resize AAHQ
python dataset_resize.py --source data/aahq-dataset/aligned --dest data/aahq-dataset/images256x256
```
**Wikiart Cityscape**: Download cityscape from [Wikiart](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan) and unzip it.

```bash
# Resize Wikiart Cityscape
python dataset_resize.py --source data/wikiart_cityscape/images --dest data/wikiart_cityscape/images256x256
```

## Train new networks using FixNoise
Using FixNoise, base command for training stylegan2-ada network as follows:

**FFHQ &rarr; MetFaces**
```bash
python train.py --outdir=${OUTDIR} --data=${DATADIR} --cfg=paper256 --resume=ffhq256 --fm=0.05
```
**FFHQ &rarr; AAHQ**
```bash
python train.py --outdir=${OUTDIR} --data=${DATADIR} --cfg=paper256 --resume=ffhq256 --fm=0.05
```
**Church &rarr; Cityscape**
```bash
python train.py --outdir=${OUTDIR} --data=${DATADIR} --cfg=stylegan2 --resume=church256 --fm=0.05
```
Additionally, we provide detailed [training scripts](./scripts/) used in our experiments.

## Inference
### Generate interpolated images
To generate interpolated images according to different noise, run:
```bash
# Generate MetFaces images without truncation
python generate.py --cfg=paper256 --outdir=out --trunc_psi=1 --seeds=865-1000 \\
    --network=pretrained/metfaces-fm0.05-001612.pkl

# Generate MetFaces images with truncation
python generate.py --cfg=paper256 --outdir=out --trunc_psi=0.7 --trunc_cutoff=8 --seeds=865-1000 \\
    --network=pretrained/metfaces-fm0.05-001612.pkl

# Generate AAHQ images with truncation
python generate.py --cfg=paper256 --outdir=out --trunc_psi=0.7 --trunc_cutoff=8 --seeds=865-1000 \\
    --network=pretrained/aahq-fm0.05-010886.pkl

# Generate Wikiart images with truncation
python generate.py --cfg=stylegan2 --outdir=out --trunc_psi=0.7 --trunc_cutoff=8 --seeds=865-1000 \\
    --network=pretrained/wikiart-fm0.05-004032.pkl
```
You can change interpolation steps by modifying `--interp-step`.

### Projecting images to latent space
To find the matching latent code for a given image file, run:
```bash
python projector_z.py --outdir=${OUTDIR} --target_dir=${DATADIR} \
    --https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl
```
We modify [projector.py](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/projector.py) to project image to z space of StyleGAN2. To use multiple gpus, add `--gpus` arguments. You can render the resulting latent vector by specifying `--projected-z-dir` for `generate.py`.
```bash
# Render an image from projected Z
python generate.py --cfg=paper256 --outdir=out --trunc_psi=0.7 --trunc_cutoff=8 \\
    --projected-z-dir=./projected --network=pretrained/aahq-fm0.05-010886.pkl
```

## Demo
We provide noise interpolation example code in [jupyter notebook](./demo.ipynb).

#### FFHQ &rarr; MetFaces
<img src="./docs/interpolation_video/metfaces/noise_interpolation_metfaces00.gif" width="45%"> &nbsp; <img src="./docs/interpolation_video/metfaces/noise_interpolation_metfaces01.gif" width="45%"> \
<img src="./docs/interpolation_video/metfaces/noise_interpolation_metfaces02.gif" width="45%"> &nbsp; <img src="./docs/interpolation_video/metfaces/noise_interpolation_metfaces03.gif" width="45%"> 

#### FFHQ &rarr; AAHQ
<img src="./docs/interpolation_video/aahq/noise_interpolation_aahq00.gif" width="45%"> &nbsp; <img src="./docs/interpolation_video/aahq/noise_interpolation_aahq01.gif" width="45%"> \
<img src="./docs/interpolation_video/aahq/noise_interpolation_aahq02.gif" width="45%"> &nbsp; <img src="./docs/interpolation_video/aahq/noise_interpolation_aahq03.gif" width="45%"> 

#### Church &rarr; Cityscape
<img src="./docs/interpolation_video/cityscape/noise_interpolation_cityscape00.gif" width="45%"> &nbsp; <img src="./docs/interpolation_video/cityscape/noise_interpolation_cityscape01.gif" width="45%"> \
<img src="./docs/interpolation_video/cityscape/noise_interpolation_cityscape02.gif" width="45%"> &nbsp; <img src="./docs/interpolation_video/cityscape/noise_interpolation_cityscape03.gif" width="45%"> 

## Citation
```
@InProceedings{Lee_2023_CVPR,
    author    = {Lee, Dongyeun and Lee, Jae Young and Kim, Doyeon and Choi, Jaehyun and Yoo, Jaejun and Kim, Junmo},
    title     = {Fix the Noise: Disentangling Source Feature for Controllable Domain Translation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14224-14234}
}
```

## License
The majority of FixNoise is licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/), however, portions of this project are available under a separate license terms: all codes used or modified from [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) is under the [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).
