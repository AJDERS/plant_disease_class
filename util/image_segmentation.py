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

def identify_leaf_segments(color_segments, segments):
    n_segments = len(np.unique(segments))
    segment_colors = np.zeros((n_segments, 3))
    green_array = np.zeros((n_segments, 3))
    green_array[:] = [0,128,0]
    found_segments = []
    for row in range(segments.shape[0]):
        for col in range(segments.shape[1]):
            # Check if we have found segment color before
            if not segments[row, col] in found_segments:
                # if not append to found segments
                found_segments.append(segments[row, col])
                # and add segment color
                segment_colors[segments[row, col]] = color_segments[row, col]
            # If all segments are found, stop.
            if len(found_segments) == n_segments:
                break

    mse_seg_green = np.sum(((segment_colors - green_array)**2), axis=1)
    leaf_seg_mask = np.ma.masked_array(mse_seg_green, mse_seg_green < 10000).mask
    leaf_seg_index = [i for (i,s) in enumerate(leaf_seg_mask) if s]
    return leaf_seg_index

def save_image(image_data):
        im = Image.fromarray(image_data.astype(np.uint8))
        im.save("test.jpeg")