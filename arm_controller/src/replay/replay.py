import h5py
from panda_py import controllers

class Replay:
    def __init__(self, arm):
        self.arm = arm

    def replay_trajectory(self, path, playback_frequency=1000):
        """
        重播指定的轨迹文件。

        参数:
        traj_name (str): HDF5 文件名称（不含路径），例如 "example_trajectory.h5"。
        path (str): 文件路径。
        playback_frequency (int): 回放频率（默认为 1000 Hz）。
        """
        h5_path = path

        # 加载 HDF5 文件中的 q 和 dq 数据
        with h5py.File(h5_path, 'r') as h5f:
            q = h5f['q'][:]
            dq = h5f['dq'][:]

        print(f"Loaded trajectory data from {h5_path}.")
        print(f"Trajectory length: {len(q)} steps.")

        # # 提示按回车键开始重播
        # input("Press Enter to replay trajectory.")

        # 将机械臂移动到初始位置（q[0]）
        self.arm.move_to_joint_position(q[0])
        print("Moved to initial joint position.")

        # 使用 JointPosition 控制器进行轨迹回放
        ctrl = controllers.JointPosition()
        self.arm.start_controller(ctrl)

        # 获取轨迹时长（通过长度和回放频率计算）
        trajectory_length = len(q)

        # 创建上下文以控制回放
        with self.arm.create_context(frequency=playback_frequency,
                                     max_runtime=2 * trajectory_length / playback_frequency) as ctx:
            i = 0
            while ctx.ok() and i < trajectory_length:
                # 每一步设置目标位置和速度
                if i % 10 == 0:
                    ctrl.set_control(q[i], dq[i])
                if i % playback_frequency == 0:  # 每秒打印一次状态
                    print(f"Replaying step {i} / {trajectory_length}")
                i += 1

        print("Replay completed.")