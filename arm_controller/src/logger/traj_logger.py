import os

import h5py

from arm_controller.src.logger.logger import Logger


class TrajectoryLogger(Logger):

    def write(self, data, path, name):
        # Save data to an HDF5 file
        q, dq = data
        h5_path = f"{path}/{name}.h5"
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset("q", data=q)
            h5f.create_dataset("dq", data=dq)

        print(f"Saved trajectory data to: {h5_path}")
