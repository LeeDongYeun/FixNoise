python train.py --outdir=./logs/aahq/ \
                --gpus=2 \
                --batch=64 \
                --kimg=12000 \
                --snap=50 \
                --data=.data/aahq-dataset/images256x256/ \
                --mirror=True \
                --cfg=paper256 \
                --resume=ffhq256 \
                --workers=2 \
                --fm=0.05 \
                --metrics=fid50k_full,kid50k_full,lpips2k