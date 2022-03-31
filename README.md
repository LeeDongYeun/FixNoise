# Fix the Noise: Disentangling Source Feature for Transfer Learning of StyleGAN</sub>

![Teaser image](./docs/figure_1.png)

**Fix the Noise: Disentangling Source Feature for Transfer Learning of StyleGAN**<br>
Dongyeun Lee, Jae Young Lee, Doyeon Kim, Jaehyun Choi, Junmo Kim<br>

Abstract: *Transfer learning of StyleGAN has recently shown great potential to solve diverse tasks, especially in domain translation. Previous methods utilized a source model by swapping or freezing weights during transfer learning, however, they have limitations on visual quality and controlling source features. In other words, they require additional models that are computationally demanding and have restricted control steps that prevent a smooth transition. In this paper, we propose a new approach to overcome these limitations. Instead of swapping or freezing, we introduce a simple feature matching loss to improve generation quality. In addition, to control the degree of source features, we train a target model with the proposed strategy, FixNoise, to preserve the source features only in a disentangled subspace of a target feature space. Owing to the disentangled feature space, our method can smoothly control the degree of the source features in a single model. Extensive experiments demonstrate that the proposed method can generate more consistent and realistic images than previous works.*

<!-- ## Requirements
* Python libraries
```bash
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 lpips
```
* Docker Users: 
Our code is highly based on the official implementation of [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). Please refer to [requirements](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements) for detailed requirements. -->