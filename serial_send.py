import serial #导入模块
import struct

class sender:
    portx="COM3"
    bps = 115200
    timex = 5
    ser = 0
    header = 'eb90'
    tail = '0d0a'
    def __init__(self, COM, baudrate = 115200, timeout = 5):
        self.portx = COM
        self.bps = baudrate
        self.timex = timeout
        # 打开串口，并得到串口对象
        self.ser = serial.Serial(self.portx, self.bps)

    def send(self, data):
        # A9是摄像头移动方案
        # body_len = len(self.header) + len(self.tail) + 1 + 4
        body_len = 'A9'
        # body_len = str(body_len)
        result = struct.pack('<f', data).hex()
        msg = self.header + body_len + result + self.tail
        msg = bytes.fromhex(msg)
        # print(msg)
        # for res in result:
        #     print(res)
        #     print(' ')
        self.ser.write(msg)

if __name__ == '__main__':
    s = sender('COM6')
    s.send(5.9)

