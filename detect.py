from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import time
import cv2
import os
import numpy as np
import datetime
import serial_send as ss
from guidance_line_algorithm.line_fitting import twoD_to_threeD, InjuryRateCal
import guidance_line_algorithm.line_fitting as line_fitting
import cam_accelerate



# from sklearn.cluster import KMeans  # sklearn库中-的cluster类中的KMeans算法
# from scipy import optimize
# import matplotlib.pyplot as plt
# import pandas as pd


zc = 960 # 相机高度 mm
# IRC = InjuryRateCal()

# temporary variable
# position = 0
# time_now = 0
# time_axis= []
# position_axis = []
# avg_time = 0
# frames = 0
# 为解决震荡问题，降低发送频率
send_every_frames = 3
curr_interval = 0

# Define pics path.
cwd = os.getcwd()
path_name = os.path.join(cwd, 'DataSet/JPEGImages')

# Define video path.
video_path = os.path.join(cwd, 'detect_result')
# Resolution (length)
input_length = 480
ratio = 640 / input_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = './checkpoints/SSD_940_9809.pth'
checkpoint = torch.load(checkpoint)  # , map_location=torch.device('cpu')
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

font = ImageFont.truetype("./calibril.ttf", 10)

sndr = ss.sender('COM6')
line_extraction = line_fitting.GuidanceLineDistractor(480, 360)

colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (255,0,255)]

file_num = 0

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    # time_prestart = time.time()

    original_image.thumbnail((input_length, input_length), Image.ANTIALIAS)
    # Transform
    image_in_nparray = np.array(original_image)
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # time_start = time.time()

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # line fitting
    # time_start = time.time()
    # 先聚类，再最小二乘拟合，得到各直线截距与斜率的平均值
    locs_list = line_extraction.create_bbox_datalist(det_boxes)
    slope, intercept, labels, lines = line_extraction.clustering_line_fitting(locs_list)

    # 计算自定义伤苗率
    # pos = [0.5, 0.5*slope+intercept]
    # IRC.isInside(boxes=det_boxes, pos=pos, label=labels)

    global position, time_now, file_num, curr_interval


    # print('frame time:\t', time_end - time_start, 's\n')

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # time_annotate = time.time()
    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    # font = ImageFont.truetype("./calibril.ttf", 10)
    # 先聚类，再最小二乘拟合，得到各直线截距与斜率的平均值
    # 画出中心点
    # for center, label in zip (locs_list, labels):
    #     # draw.point([(center[0], center[1])], colors[label])
    #     draw.ellipse((center[0] * original_image.size[0] - 10, center[1] * original_image.size[1] - 10, center[0] * original_image.size[0] + 10, center[1] * original_image.size[1] + 10), fill=colors[label], outline=colors[label], width=2)
    #
    # for line in lines:
    #     draw.line([(0, line[1] * original_image.size[1]),
    #                (original_image.size[0], line[1] * original_image.size[1] + line[0] * original_image.size[0])],
    #               fill=(255, 255, 255), width=3)
    #
    draw.line([(0, intercept * original_image.size[1]),
               (original_image.size[0], intercept * original_image.size[1] + slope * original_image.size[0])],
              fill=(0, 255, 255), width=3)

    ret = calculate_pos(slope, intercept, ratio, original_image.size[0], original_image.size[1])
    curr_interval += 1
    if curr_interval % send_every_frames == 0:
        curr_interval = 0
        sndr.send(float(ret))  # 发送当前位置

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    # time_end = time.time()
    # print('frame time:\t', time_end - time_start, 's\r\n', 'frame time(including pretreatment):\t',
    #       time_end - time_prestart, 's\r\n', 'annotate_time:\t', time_end - time_annotate)

    # if len(lines) == 2:
    #     img_with_lines = np.array(annotated_image, dtype=np.uint8)
    #     img_with_lines = Image.fromarray(img_with_lines.astype(np.uint8))
    #     # 保存图片
    #     # global file_num
    #     img_with_lines.save(os.path.join('image/lines=2', str(file_num) + '.jpg'))
    #
    # elif len(lines) == 1:
    #     img_with_lines = np.array(annotated_image, dtype=np.uint8)
    #     img_with_lines = Image.fromarray(img_with_lines.astype(np.uint8))
    #     # 保存图片
    #     # global file_num
    #     img_with_lines.save(os.path.join('image/lines=1', str(file_num) + '.jpg'))
    #
    # else:
    #     img_with_lines = np.array(annotated_image, dtype=np.uint8)
    #     img_with_lines = Image.fromarray(img_with_lines.astype(np.uint8))
    #     # 保存图片
    #     # global file_num
    #     img_with_lines.save(os.path.join('image/lines=3', str(file_num) + '.jpg'))
    # file_num += 1

    return annotated_image


def video_detect(file, cover_path, fps=30, timeF=1):
    global avg_time, frames
    # file = os.path.join(path, item)
    cap = cam_accelerate.camCapture(file)
    cap.start()
    cv2.waitKey(1000)
    # cap = cv2.VideoCapture(file)  # 按照绝对路径打开视频
    cover_path = os.path.abspath(cover_path)



    # Define the codec and create VideoWriter object
    if cap.isOpened():
        # cap.set(1,x) 设置要解码/捕获的帧的基于0的索引
        # cap.set(1, int(cap.get(7) / 2))  # 取它的中间帧,cap.get(7)获取视频总帧数
        # cap.read()返回两个参数赋给两个值。
        # 第一个参数rval的值为True或False，代表有没有读到图片。
        # 第二个参数是frame，是当前截取一帧的图片。
        # cap.set(3, 480)  # width=1920
        # cap.set(4, 270)  # height=1080
        frame = cap.getframe()

        frame = Image.fromarray(frame)
        frame.thumbnail((input_length, input_length))

        sp = frame.size
        sz = (sp[0], sp[1])
        f = 1
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(os.path.join(cover_path, f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.avi'), fourcc, fps, sz)
        # total time consumed


        step = 6
        while True:  # 循环读取视频帧
            time_start = time.time()
            frame = cap.getframe()
            if frame is None:
                break
            if f % timeF == 0:  # 每隔timeF帧进行存储操作
                if f % step != 0:
                    f = f + 1
                    continue
                # cv2.imshow('test', frame)
                frame = Image.fromarray(frame)
                frame = detect(frame, min_score=0.3, max_overlap=0.45, top_k=200)
                frame = np.array(frame)
                out.write(frame)
                cv2.imshow('frame', frame)
            f = f + 1
            cv2.waitKey(1)
            time_end = time.time()
            time_interval = time_end - time_start

            # avg_time += time_interval
            # frames += 1
            print('frame time:\t', time_interval, 's\r\n')

        cap.stop()  # 关闭打开的文件
        # print('avg_time = ', avg_time / frames)
        # out.release()



def calculate_pos(slope, intercept, ratio, length, height):
    pos_x = 0.5 * length * ratio
    pos_y = slope * pos_x + intercept * height * ratio
    pos_3d = [pos_x, pos_y, 1]
    origin_pos_3d = [0.5 * length * ratio, 0.5 * height * ratio, 1]
    pos = twoD_to_threeD(pos_3d, zc)
    origin_3d = twoD_to_threeD(origin_pos_3d, zc)
    offset = pos[1] - origin_3d[1]

    # offset += 22

    print(offset)
    if abs(offset) < 10:
        return 0
    else:
        return offset


if __name__ == '__main__':
    # img_path = 'DataSet/JPEGImages/000021.jpg'
    # original_image = Image.open(img_path, mode='r')
    # original_image = original_image.convert('RGB')
    # detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
    # plt.ion()

    video_file = 'D:\\华农项目资料\\深度学习\\raw\\2020_01_06_17_27_IMG_0869.MOV'
    output_path = os.path.join(video_path, 'output')
    video_detect(1, output_path, 15, 1)

    # IRC.calculateInjuryRate()

