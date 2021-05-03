import os
import time
import configparser
from picamera import PiCamera

class Camera():
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.camera = PiCamera()

    def take_picture(self, output_directory):
        self.camera.capture(output_directory)
        


