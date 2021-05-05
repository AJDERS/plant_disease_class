import numpy as np
import skimage.color as color
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import slic

def _load_image(image_dir):
    image = Image.open(image_dir)
    image_rgb = image.convert('RGB')
    image_array = np.asarray(image_rgb)
    return image_array

def slic_segment(image_dir, n_segments):
    image = _load_image(image_dir)
    segments = slic(image, n_segments)
    color_segments = color.label2rgb(segments, image, kind='avg')
    save_image(color_segments)
    return color_segments, segments

def save_image(image_data):
        im = Image.fromarray(image_data.astype(np.uint8))
        im.save("test.jpeg")