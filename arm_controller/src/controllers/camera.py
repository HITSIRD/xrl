import threading
import time
import pyrealsense2 as rs
import numpy as np


class Camera:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.width = width
        self.height = height
        self.fps = fps

        self.started = False
        self.start()

    def start(self):
        try:
            self.pipeline.start()
            self.started = True
            print("[INFO] RealSense camera starting.")
            time.sleep(5)  # 给摄像头时间完成初始化
        except RuntimeError as e:
            print(f"[ERROR] Failed to start RealSense pipeline: {str(e)}")
            self.started = False
            raise

    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("[WARNING] No color frame received!")
                return None
            color_image = np.asanyarray(color_frame.get_data())
            return color_image
        except RuntimeError as e:
            print(f"[ERROR] Failed to retrieve frame: {str(e)}")
            return None

class CameraThread(threading.Thread):
    def __init__(self, width=640, height=480, fps=30):
        super(CameraThread, self).__init__()
        self.camera = Camera(width, height, fps)
        self.frame = None
        self.running = True

    def run(self):
        print("[INFO] Camera thread started.")
        while self.running:
            # Continuously get the latest frame
            import home.views
            home.views.frame = self.camera.get_frame()
            # if frame is not None:
            #     self.frame = frame
            # time.sleep(0.01)  # Slight delay to reduce CPU usage

    def stop(self):
        print("[INFO] Stopping camera thread.")
        self.running = False
        try:
            self.quit()
        except Exception as e:
            print(f"[WARNING] Error while stopping camera thread: {e}")