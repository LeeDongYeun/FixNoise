# Fix the Noise: Disentangling Source Feature for Transfer Learning of StyleGAN</sub>

![Teaser image](./docs/figure_1.png)
**Fix the Noise: Disentangling Source Feature for Transfer Learning of StyleGAN**<br>
Dongyeun Lee, Jae Young Lee, Doyeon Kim, Jaehyun Choi, Junmo Kim<br>

>**Abstract**: 
*Transfer learning of StyleGAN has recently shown great potential to solve diverse tasks, especially in domain translation. Previous methods utilized a source model by swapping or freezing weights during transfer learning, however, they have limitations on visual quality and controlling source features. In other words, they require additional models that are computationally demanding and have restricted control steps that prevent a smooth transition. In this paper, we propose a new approach to overcome these limitations. Instead of swapping or freezing, we introduce a simple feature matching loss to improve generation quality. In addition, to control the degree of source features, we train a target model with the proposed strategy, FixNoise, to preserve the source features only in a disentangled subspace of a target feature space. Owing to the disentangled feature space, our method can smoothly control the degree of the source features in a single model. Extensive experiments demonstrate that the proposed method can generate more consistent and realistic images than previous works.*

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
| FFHQ &rarr; MetFaces    |    256x256    | paper256 | Trained initialized with official [pre-trained model on FFHQ 256](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/) from Pytorch implementation of [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). |
| FFHQ &rarr; AAHQ        |    256x256    | paper256 | Trained initialized with official [pre-trained model on FFHQ 256](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/) from Pytorch implementation of [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). |
| Church &rarr; Cityscape |    256x256    | stylegan2| Trained initialized with official [pre-trained model on LCUN Church config-f](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/) from Tensorflow implementation of [stylegan2](https://github.com/NVlabs/stylegan2). |

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

**MetFaces**
```bash
python train.py --outdir=${OUTDIR} --data=${DATADIR} --cfg=paper256 --resume=ffhq256 --fm=0.05
```
**AAHQ**
```bash
python train.py --outdir=${OUTDIR} --data=${DATADIR} --cfg=paper256 --resume=ffhq256 --fm=0.05
```
**Wikiart Cityscape**
```bash
python train.py --outdir=${OUTDIR} --data=${DATADIR} --cfg=stylegan2 --resume=church256 --fm=0.05
```
Additionally, we provide detailed [training scripts](./scripts/) used in our experiments.