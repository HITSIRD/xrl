import os
import h5py


class Logger():
    def __init__(self):
        pass

    def write(self, data, file_name):
        # write_path = os.path.join(self.log_dir, file_name)
        # print(f"writing to {write_path}")
        #
        # with h5py.File(write_path, "w") as f:
        #     f.create_dataset("data", data=data)
        raise NotImplementedError


    def read(self):
        raise NotImplementedError
