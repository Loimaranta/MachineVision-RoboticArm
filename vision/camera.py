import cv2

class Camera:
    def __init__(self, camera_index: int=0) -> None:
        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

    def get_frame(self):
        success, frame = self.camera.read()

        if not success:
            raise RuntimeError("Could not read frame")

        return frame

    def release(self) -> None:
        self.camera.release()
