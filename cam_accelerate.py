import threading
import cv2

class camCapture:
    def __init__(self, dev):
        self.Frame = 0
        self.status = False
        self.isstop = False

        # 摄影机连接。
        self.capture = cv2.VideoCapture(dev)

    def isOpened(self):
        return self.capture.isOpened()

    def start(self):
        # 把程序放进子线程，daemon=True 表示该线程会随着主线程关闭而关闭。
        print('cam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        # 记得要设计停止无限循环的开关。
        self.isstop = True
        print('cam stopped!')

    def getframe(self):
        # 当有需要影像时，再回传最新的影像。
        return self.Frame

    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()

        self.capture.release()