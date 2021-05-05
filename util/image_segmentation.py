import numpy as np
import skimage.color as color
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from skimage.segmentation import slic
from skimage.color import deltaE_cie76

def _load_image(image_dir):
    image = Image.open(image_dir)
    image_rgb = image.convert('RGB')
    image_array = np.asarray(image_rgb)
    return image_array

def slic_segment(image_dir, n_segments):
    image = _load_image(image_dir)
    segments = slic(image, n_segments, compactness=25)
    color_segments = color.label2rgb(segments, image, kind='avg')
    save_image(color_segments, 'color_segments')
    save_image(segments, 'segments')
    return color_segments, segments

def identify_leaf_segments(color_segments, segments):
    n_segments = len(np.unique(segments))
    segment_colors = np.zeros((n_segments, 3))
    green_array = np.array([0,128,0])
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

    mse_seg_green = np.apply_along_axis(lambda x: deltaE_cie76(x, green_array), -1, segment_colors) 
    # MSE of 10000 between segment color and green is experimentally set
    leaf_seg_mask = np.ma.masked_array(mse_seg_green, mse_seg_green < 120).mask
    leaf_seg_color = [segment_colors[i] for (i,s) in enumerate(leaf_seg_mask) if s]
    return leaf_seg_color

def minimum_bounding_box(color_segments, leaf_seg_color):
    bounding_boxes = []
    for leaf_color in leaf_seg_color:
        leaf_coords = np.argwhere(color_segments == leaf_color)
        min_x, min_y = np.min(leaf_coords[...,0]), np.min(leaf_coords[...,1])
        max_x, max_y = np.max(leaf_coords[...,0]), np.max(leaf_coords[...,1])
        bounding_boxes.append([min_x, min_y, max_x, max_y])
    return bounding_boxes

def add_bounding_boxes(image_dir, bounding_boxes, name):
    im = Image.open(image_dir)
    fig, ax = plt.subplots()
    ax.imshow(im)

    for (min_x, min_y, max_x, max_y) in bounding_boxes:
        # Create a Rectangle patch
        diff_x = max_x - min_x
        diff_y = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), diff_x, diff_y, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig(f'test_{name}.jpeg')



def save_image(image_data, name):
        im = Image.fromarray(image_data.astype(np.uint8))
        im.save(f"{name}.jpeg")