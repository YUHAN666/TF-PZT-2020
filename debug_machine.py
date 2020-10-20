from PIL import Image
import numpy as np
import cv2
import pathlib
import os

def concatImage(images, mode="L"):
    if not isinstance(images, list):
        raise Exception('images must be a  list  ')
    count = len(images)
    size = Image.fromarray(images[0]).size
    target = Image.new(mode, (size[0] * count, size[1] * 1))
    for i in range(count):
        image = Image.fromarray(images[i]).resize(size, Image.BILINEAR)
        target.paste(image, (i*size[0], 0, (i+1)*size[0], size[1]))
    return target


image_root = 'E:/CODES/FAST-SCNN/DATA/surplus2/val_image/'
mask_path = 'E:/CODES/FAST-SCNN/DATA/surplus2/val_mask/'
save_dir = 'D:/123/'

# image_path = pathlib.Path(image_path)
mask_path = pathlib.Path(mask_path)

# image_path = list(image_path.glob('*'))
mask_path = list(mask_path.glob('*'))

# image_path = [str(path) for path in image_path]
mask_path = [str(path) for path in mask_path]

for i in range(len(mask_path)):
    filename = mask_path[i].split('\\')[-1]
    image_path = image_root+filename


    mask = cv2.imread(mask_path[i], 0)
    mask = np.array(mask)
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
    image = np.array(image)

    # mask = mask*255
    vis = np.where(mask>0, 255, image)

    # filename = image_path[i].split('\\')[-1]

    image_list = [image, mask, vis]
    img_visual = concatImage(image_list)
    visualization_path = os.path.join(save_dir, filename)
    img_visual.save(visualization_path)