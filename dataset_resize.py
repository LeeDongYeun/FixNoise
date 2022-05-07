import os
import click
import numpy as np
from PIL import Image

IMG_EXTENSIONS = [
    'jpg', 'jpeg', 'png', 'ppm', 'bmp',
    'pgm', 'tif', 'tiff', 'webp',
    'JPG', 'JPEG', 'PNG', 'PPM', 'BMP',
    'PGM', 'TIF', 'TIFF', 'WEBP'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

@click.command()
@click.pass_context
@click.option('--source', help='Directory for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory for output dataset', required=True, metavar='PATH')
@click.option('--width', help='Output width', default=256, type=int)
@click.option('--height', help='Output height', default=256, type=int)
@click.option('--crop', help='Whether to crop or not', default=False, type=bool)
def resize_dataset(ctx, source, dest, width, height, crop):
    """Resize dataset
    Ex)
        --source data/metfaces/images
        --dest data/metfaces/images256x256
        
        --source data/aahq-dataset/aligned
        --dest data/aahq-dataset/images256x256

        --source data/wikiart_cityscape/images
        --dest data/wikiart_cityscape/images256x256
    """
    if not os.path.exists(dest):
        os.makedirs(dest)

    image_paths = make_dataset(source)
    print(len(image_paths))
    i = 0
    for img_path in image_paths:
        img = np.array(Image.open(img_path))
        if crop:
            crop_size = np.min(img.shape[:2])
            img = img[(img.shape[0] - crop_size) // 2 : (img.shape[0] + crop_size) // 2, (img.shape[1] - crop_size) // 2 : (img.shape[1] + crop_size) // 2]
        img = Image.fromarray(img, 'RGB')
        img = img.resize((width, height), Image.BILINEAR)

        img_name = img_path.split('/')[-1]
        new_img_path = os.path.join(dest, img_path.split('/')[-2])

        if not os.path.exists(new_img_path):
            os.makedirs(new_img_path)
        
        img.save(os.path.join(new_img_path, img_name))

        if i % 1000 == 0:
            print(i, "images done")
        i += 1

    print("Resize Done!!!")


if __name__ == '__main__':
	resize_dataset()