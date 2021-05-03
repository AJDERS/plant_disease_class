import os
import time
import sched
import configparser
import shutil
from compiler import Compiler
from camera.camera import Camera

class Scheduler():

    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.start_time = time.time()
        self.picture_interval = self.config['SCHEDULE'].getfloat('PictureInterval')
        self.runtime = self.config['SCHEDULE'].getfloat('ScheduleRunTime')
        self.w_capture = self.config['SCHEDULE'].get('WithCapture')
        self.w_prediction = self.config['SCHEDULE'].get('WithPrediction')
        self.buffer_time = self.config['SCHEDULE'].getint('PictureBufferTime')
        self.camera_output_dir = self.config['CAMERA'].get('CameraOutputDirectory')
        self.compiler = Compiler(config_path)
        self.camera = Camera(config_path)
        self.schedule = sched.scheduler(time.time, time.sleep)

    def _make_output_dir(self, t):
        timestamp = self.start_time + t
        formatted_ts = time.strftime('%H:%M:%S', time.gmtime(float(timestamp)))
        output_dir = os.path.join(self.camera_output_dir, f'{formatted_ts}.jpg')
        return output_dir

    def build_schedule(self):
        if self.w_capture == 'Y' and self.w_prediction == 'Y':
            for t in range(1, int(self.runtime)+1, int(self.picture_interval)):
                output_dir = self._make_output_dir(t)
                self.schedule.enterabs(
                    self.start_time + t,
                    1,
                    self.camera.take_picture,
                    argument = (output_dir,)
                )
                self.schedule.enterabs(
                    self.start_time + t + self.buffer_time,
                    1,
                    self.compiler.predict,
                    argument = (output_dir,)
                )

        elif self.w_capture == 'Y' and self.w_prediction == 'N':
            for t in range(1, int(self.runtime)+1, int(self.picture_interval)):
                output_dir = self._make_output_dir(t)
                self.schedule.enterabs(
                    self.start_time + t,
                    1,
                    self.camera.take_picture,
                    argument = (output_dir,)
                )

        elif self.w_capture == 'N' and self.w_prediction == 'Y':
            for t in range(1, int(self.runtime)+1, int(self.picture_interval)):
                output_dir = self._make_output_dir(t)
                self.schedule.enterabs(
                    self.start_time + t + self.buffer_time,
                    1,
                    self.compiler.predict,
                    argument = (output_dir,)
                )

        else:
            print('Neither prediction or capture is toggled.')

    def run(self):
        self.build_schedule()
        self.schedule.run()
        
