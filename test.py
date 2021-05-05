import os 
from util.scheduler import Scheduler
from util import image_segmentation


def main():
    #s = Scheduler('default_config.ini')
    #s.run()
    color_segments, segments = image_segmentation.slic_segment('captured_images/2021:05:05:08:44:55.jpg', 25)
    leaf_seg_index = image_segmentation.identify_leaf_segments(color_segments, segments)
    bounding_boxes = image_segmentation.minimum_bounding_box(color_segments, leaf_seg_index)
    image_segmentation.add_bounding_boxes('captured_images/2021:05:05:08:44:55.jpg', bounding_boxes, 'im')
    image_segmentation.add_bounding_boxes('color_segments.jpeg', bounding_boxes, 'segs')

if __name__=='__main__':
    main()