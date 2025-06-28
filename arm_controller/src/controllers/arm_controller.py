import time

import numpy as np
from panda_py import controllers

class ArmController:
    def __init__(self, arm):
        self.arm = arm

    def move_to(self, joint_position):
        self.arm.move_to_joint_position(joint_position)

    def get_pose(self):
        # self.arm.teaching_mode(True)
        # time.sleep(0.01)
        # self.arm.teaching_mode(False)
        return self.arm.get_position(), self.arm.get_orientation()

    def get_state(self):
        return self.arm.get_state()

    def create_controller(self):
        return controllers.JointPosition()

    def create_context(self, length):
        return self.arm.create_context(frequency=1000, max_runtime=length)