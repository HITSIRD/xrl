from arm_controller.src.logger.logger import Logger


class PosLogger(Logger):
    def __init__(self, arm, log_dir):
        self.arm = arm
        self.log_dir = log_dir

    def read(self):
        pos = self.arm.q.tolist()
        print(f"Current arm pos: {pos}")
        return pos
        #  TODO

    # def write(self, data, file_name):
    #     write_path = os.path.join(self.log_dir, file_name)
    #     print(f"writing to {write_path}")
    #
    #     with h5py.File(write_path, "w") as f:
    #         f.create_dataset("data", data=data)

# import panda_py
#
# arm = panda_py.Panda(ARM_URL)
# poslog = PosLogger(arm, 'test')
# q = poslog.read()
# arm.move_to_start()
# arm.move_to_joint_position(q)
