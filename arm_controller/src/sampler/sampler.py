import cv2
import numpy as np
import h5py
import keyboard
import time


class Sampler():
    def __init__(self, arm, camera):
        self.arm = arm
        self.camera = camera

    def sample_pos(self):
        self.arm.teaching_mode(True)
        #print("Teaching mode on")
        time.sleep(0.01)
        self.arm.teaching_mode(False)
        pos = self.arm.get_state().q
        # pos = self.arm.get_position()
        # orientation = self.arm.get_orientation()
        print(f"Current arm pos: {pos}")
        return pos

    def sample_trajectory(self):
        print("Press 's' to start recording the trajectory and 'e' to end.")

        # Wait for the 's' key to start logging
        input("按下 Enter 键开始录制...")

        print("Starting trajectory recording...")
        self.arm.teaching_mode(True)

        # Set a very large logging duration (e.g., 1e5 steps) to accommodate dynamic length
        self.arm.enable_logging(100000)
        start_time = time.time()

        # Wait for the 'e' key to stop logging
        input("按下 Enter 键结束录制...")

        self.arm.teaching_mode(False)
        elapsed_time = time.time() - start_time
        print(f"Trajectory recording stopped. Duration: {elapsed_time:.2f} seconds.")

        # Fetch logged data
        q = self.arm.get_log()['q']
        dq = self.arm.get_log()['dq']

        return q, dq
