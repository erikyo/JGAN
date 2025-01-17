import os
import matplotlib.pyplot as plt
import numpy as np

from model.srgan import generator

from model import resolve_single
from utils import load_image
from PIL import Image

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

pre_generator = generator()
gan_generator = generator()

weights_dir = 'weights\\srgan'
images_dir = '.images'
weights_file = lambda filename: os.path.join(weights_dir, filename)

pre_generator.load_weights(weights_file('pre_generator.h5'))
gan_generator.load_weights(weights_file('gan_generator.h5'))


def sliceImage(image, tiles_x_size=48, tiles_y_size=48):
    return [image[x:x + tiles_x_size, y:y + tiles_y_size] for x in range(0, image.shape[0], tiles_x_size) for y in
            range(0, image.shape[1], tiles_y_size)]

def tile_count(side,tile_size):
    return round(side/tile_size) + 1

def get_tile_size(tile_size, resize_ratio):
    return int( tile_size / resize_ratio )

def resolve_and_plot(lr_image_path, tile_size=48, scale=4):

    lr = Image.open(lr_image_path)
    width, height = lr.size
    x_tiles, y_tiles = tile_count(width,tile_size), tile_count(height,tile_size)
    print("Image size", width, "x", height)
    print("Image split into", x_tiles*y_tiles,":", x_tiles, "X tiles and", y_tiles, "Y (", tile_size, "px tiles)")
    lr_slices = sliceImage(np.array(lr), tile_size, tile_size)

    processed_slices = [];
    increment = 0

    for slice in lr_slices:
        pre_sr = resolve_single(pre_generator, slice)
        gan_sr = resolve_single(gan_generator, slice)

        pil_img = tf.keras.preprocessing.image.array_to_img(gan_sr)
        processed_slices.append(pil_img)

    new_im = Image.new('RGB', (width * scale, height * scale))
    w, h = new_im.size
    index = 0

    tile_width, tile_height = processed_slices[0].size;
    resize_ratio = tile_width / tile_size;

    for j in range(0, h, tile_size * scale):
        for i in range(0, w, tile_size * scale):

            tile_width, tile_height = processed_slices[index].size;
            new_im.paste(processed_slices[index].resize( (get_tile_size(tile_width, resize_ratio), get_tile_size(tile_height, resize_ratio) ), Image.Resampling.LANCZOS), (i, j))
            index = index + 1

    filename= os.path.splitext(lr_image_path)[0] + "-res" + ".png"
    new_im.save(filename)

resolve_and_plot("demo/0869x4-crop.png", tile_size=40, scale=1)