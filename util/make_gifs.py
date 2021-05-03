from PIL import Image
import glob
 
def make_gif(path):
    # Create the frames
    frames = []
    imgs = glob.glob(f"{path}/*.jpg")
    for_use = sorted(imgs, key=lambda x: float(''.join(x.split('/')[-1].split('.')[:-1])))
    for i in for_use:
        new_frame = Image.open(i)
        frames.append(new_frame)
    
    # Save into a GIF file that loops forever
    frames[0].save(f'{path}/time_lapse.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100, loop=0)