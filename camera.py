import os
import time
from picamera import PiCamera

class Camera():
    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.camera = PiCamera()

    def take_picture(self):
        self.camera.capture(os.path.join(self.output_directory, f'{time.time()}.jpg'), resize=(256, 192))


