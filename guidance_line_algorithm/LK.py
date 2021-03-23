import numpy as np
import cv2
import time

video_file = 'D:\\华农项目资料\\深度学习\\raw\\2020_01_06_17_28_IMG_0870.MOV'
cap = cv2.VideoCapture(video_file)


# ShiTomasi 角点检测参数
feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# lucas kanade光流法参数
lk_params = dict( winSize  = (150,150),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建随机颜色
color = np.random.randint(0,255,(100,3))

# 获取第一帧，找到角点
ret, old_frame = cap.read()
#找到原始灰度图
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#获取图像中的角点，返回到p0中
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# 创建一个蒙版用来画轨迹
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    time_start = time.time()
    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    time_end = time.time()
    print(time_end - time_start, "s/frame.\n")
    # 选取好的跟踪点
    good_new = p1[st==1]
    good_old = p0[st==1]

    # 画出轨迹
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # 更新上一帧的图像和追踪点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()