import os
import time
import threading

import h5py
import numpy as np

from arm_controller.src.controllers.arm_controller import ArmController
from arm_controller.src.controllers.hand_controller import HandController

class DataRecorder(threading.Thread):
    def __init__(self, arm, hand, camera, save_path, save_interval=1 / 10, callback=None):
        super().__init__()
        self.arm = ArmController(arm)
        self.hand = HandController(hand)
        self.camera = camera
        self.save_path = save_path
        self.save_interval = save_interval
        self.running = threading.Event()

        self.q_data = []
        self.pose_data = []
        self.force_data = []
        self.tactile_data = []
        self.image_obs_data = np.zeros((1800, self.camera.height, self.camera.width, 3), dtype=np.uint8)
        self.skill_label = []
        self.callback = callback  # Optional callback for when data is saved
        self.frame_count = 0
        self.skill = None

    def update_skill(self, skill):
        self.current_skill = skill

    def run(self):
        self.running.set()

        while self.running.is_set():
            import views
            if home.views.skill_label is not None:
                self.skill = home.views.skill_label

            if self.skill is not None:
                timestamp = time.time()
                # 采集运动学与图像数据
                position, orientation = self.arm.get_pose()
                q = self.arm.get_state().q
                widths = [self.hand.read_width()]
                # 存储数据
                self.pose_data.append([*position, *orientation, *widths])
                self.q_data.append(q)
                # save skill lable
                self.skill_label.append(self.skill)
                # print("appended")
                image_obs = self.camera.get_frame()
                self.image_obs_data[self.frame_count] = image_obs
                self.frame_count += 1
                process_time = time.time() - timestamp
                # 动态调整睡眠时间，确保总间隔接近 save_interval
                print(process_time)
                sleep_time = max(0.0, self.save_interval - process_time)
                time.sleep(sleep_time)

        # 线程结束时保存数据
        self.save_data()

    def stop_and_save(self):
        self.running.clear()
        self.join()  # 等待 run() 方法结束

    def save_data(self):
        os.makedirs(self.save_path, exist_ok=True)
        print("Saving data...")
        print(f"sum frame count: {self.frame_count}")

        with h5py.File(os.path.join(self.save_path, 'traj.h5'), 'w') as h5f:
            h5f.create_dataset('pose', data=np.array(self.pose_data))
            h5f.create_dataset('q', data=np.array(self.q_data))
            h5f.create_dataset('skill', data=np.array(self.skill_label, dtype=h5py.string_dtype(encoding='utf-8')))
            h5f.create_dataset('rgb', data=np.array(self.image_obs_data[:self.frame_count]))

        # 若需要：保存力或触觉数据
        # np.save(os.path.join(self.save_path, 'force.npy'), np.array(self.force_data))
        # np.save(os.path.join(self.save_path, 'tactile.npy'), np.array(self.tactile_data))

        if self.callback:
            self.callback()
