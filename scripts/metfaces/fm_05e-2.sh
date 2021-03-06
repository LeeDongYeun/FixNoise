python train.py --outdir=./logs/metfaces/ \
                --gpus=2 \
                --batch=64 \
                --kimg=2000 \
                --snap=10 \
                --data=.data/metfaces/images256x256/ \
                --mirror=True \
                --cfg=paper256 \
                --resume=ffhq256 \
                --workers=2 \
                --fm=0.05 \
                --metrics=fid50k_full,kid50k_full,lpips2k