import os
from os import listdir, path
from os.path import isfile, join
import time
import datetime
import sched
import configparser
import shutil
from compiler import Compiler
from camera.camera import Camera
from util.save_to_csv import record_is_in_csv

class Scheduler():

    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.instantiating_time = time.time()
        self.picture_interval = self.config['SCHEDULE'].getfloat('PictureInterval')
        self.start_time = self.convert_time_str(self.config['SCHEDULE'].get('StartTime'))
        self.end_time = self.convert_time_str(self.config['SCHEDULE'].get('EndTime')) - self.start_time
        self.w_capture = self.config['SCHEDULE'].get('WithCapture')
        self.w_prediction = self.config['SCHEDULE'].get('WithPrediction')
        self.buffer_time = self.config['SCHEDULE'].getint('PictureBufferTime')
        self.camera_output_dir = self.config['CAMERA'].get('CameraOutputDirectory')
        self.prediction_output_dir = self.config['PREDICTION'].get('PredictionOutputDirectory')
        self.compiler = Compiler(config_path)
        self.camera = Camera(config_path)
        self.schedule = sched.scheduler(time.time, time.sleep)

    def convert_time_str(self, string):
        return time.mktime(time.strptime(string, '%Y:%m:%d:%H:%M:%S'))

    def _make_output_dir(self, t):
        timestamp = self.start_time + t
        formatted_ts = time.strftime('%Y:%m:%d:%H:%M:%S', time.localtime(float(timestamp)))
        output_dir = os.path.join(self.camera_output_dir, f'{formatted_ts}.jpg')
        return output_dir

    def get_previous_captured_unpredicted(self, start_timestamp):
        files = [f for f in listdir(self.camera_output_dir) if isfile(join(self.camera_output_dir, f))]
        timestamps = [f.split('.')[0] for f in files]
        #start_time = time.strptime(start_timestamp, '%Y:%m:%d:%H:%M:%S')
        previous_captured_unpredicted = []
        start_timestruct = time.localtime(start_timestamp)
        for i,timestamp in enumerate(timestamps):
            t = time.strptime(timestamp, '%Y:%m:%d:%H:%M:%S')
            # check if captured image are taken before start time.
            if max((start_timestruct,t)) == start_timestruct:
                # check records if we have already predicted.
                if not record_is_in_csv(self.prediction_output_dir, timestamp):
                    previous_captured_unpredicted.append(files[i])
        return previous_captured_unpredicted

    def predict_previous_captures(self):
        timestamp = self.start_time
        previous_captured_unpredicted = self.get_previous_captured_unpredicted(timestamp)
        for file in previous_captured_unpredicted:
            self.compiler.predict(os.path.join(self.camera_output_dir,file))

    def build_schedule(self):
        if self.w_capture == 'Y' and self.w_prediction == 'Y':
            for t in range(0, int(self.end_time), int(self.picture_interval)):
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
            self.predict_previous_captures()

        elif self.w_capture == 'Y' and self.w_prediction == 'N':
            for t in range(0, int(self.end_time), int(self.picture_interval)):
                output_dir = self._make_output_dir(t)
                self.schedule.enterabs(
                    self.start_time + t,
                    1,
                    self.camera.take_picture,
                    argument = (output_dir,)
                )

        elif self.w_capture == 'N' and self.w_prediction == 'Y':
            self.predict_previous_captures()

        else:
            print('Neither prediction or capture is toggled.')

    def run(self):
        self.build_schedule()
        self.schedule.run()
        
