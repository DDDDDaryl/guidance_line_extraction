import numpy as np
from sklearn.cluster import KMeans  # sklearn库中的cluster类中的KMeans算法
from scipy import optimize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import queue
# from guidance_line_algorithm import vis_3d as vis

# 相机参数
weeder_camera_intrinsic = {
    # R，旋转矩阵

    "R": [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]],
    # t，平移向量
    "T": [0, 0, 1500.0000000],
    # 焦距，f/dx, f/dy
    "f": [1.4765e3, 1.4749e3],
    # principal point，主点，主轴与像平面的交点
    "c": [656.7926, 365.5874],
    "IntrinsicMatrix": [[1.4765e3, 0, 0.6568e3, 0],
                        [0, 1.4749e3, 0.3656e3, 0],
                        [0, 0, 0.0010e3, 0]],
    "AspectRatio": 16/9
}

exp_camera_intrinsic = {
    # R，旋转矩阵

    "R": [[0.9982, 0.0363, 0.0478],
          [-0.0373, 0.9991, 0.0189],
          [-0.0471, -0.0206, 0.9987]],
    # t，平移向量
    "T": [-63.9398, -61.9131, 579.8560],
    # 焦距，f/dx, f/dy
    "f": [1.4233e3, 1.4220e3],
    # principal point，主点，主轴与像平面的交点
    "c": [660.4364, 371.8500],
    "IntrinsicMatrix": [[1.4233e3, 0, 660.4364, 0],
                        [0, 1.4220e3, 371.8500, 0],
                        [0, 0, 1, 0]],
    "AspectRatio": 16/9
}

camera_intrinsic = exp_camera_intrinsic
#

class GuidanceLineDistractor:
    def __init__(self, image_w, image_h, method='DBSCAN'):
        self.state = False
        self.image_width = image_w
        self.image_height = image_h
        self.method_DBSCAN = dict(eps=0.15,  # 邻域半径
                                    min_samples=3,  # 最小样本点数，MinPts
                                    metric='euclidean',
                                    metric_params=None,
                                    algorithm='auto',
                                    # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点
                                    leaf_size=30,  # balltree,cdtree的参数
                                    p=None,  #
                                    n_jobs=1)
        self.method_KM = dict(n_clusters=2)  # 聚类中心
        self.method = method
        self.slope_window_size = 5
        self.slope_window = queue.Queue(self.slope_window_size)
        self.intercept_window_size = 5
        self.intercept_window = queue.Queue(self.intercept_window_size)
        self.smoothed_slope = 0
        self.smoothed_intercept = 0.5
        self.lines = []
        self.opt_pos = 0  # 最优位置
        self.intercept_window.put(0.5)
        self.intercept_window.put(0.5)
        self.intercept_window.put(0.5)
        self.intercept_window.put(0.5)
        self.intercept_window.put(0.5)


    def create_bbox_datalist(self, boxes):
        """
        Accept a tensor of detected bounding boxes and transform into lists.

        :param boxes: bounding boxes in tensor
        :return: lists of x and y coordinates, respectively
        """
        bboxes_list = boxes[0].tolist()
        # normalized coords
        locs_list = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox
                     in bboxes_list]
        return locs_list

    def clustering_line_fitting(self, locs_list, threshold=3):
        """
            Clustering and perform line fitting algorithm.

            :param locs_list: list of bbox coordinates
            :return: lines represented in (k, b) which means y=kx+b
        """
        # locs_list = self.create_bbox_datalist(boxes)
        lines = []

        if len(locs_list) < threshold:
            return self.smoothed_slope, self.smoothed_intercept, [], lines

        x = np.array([i[0] for i in locs_list]).reshape(-1, 1)
        y = np.array([i[1] for i in locs_list]).reshape(-1, 1)

        # 投影到像素空间的y轴，再做聚类
        if self.method == 'DBSCAN':
            clustering = DBSCAN(**self.method_DBSCAN).fit(y)
        elif self.method == 'KM':
            clustering = KMeans(**self.method_KM).fit(y)
        else:
            print("usage: \'DBSCAN\' or \'KM\' \n")
            exit(-1)

        class_num = max(clustering.labels_) + 2
        ret = [[] for i in range(class_num)]

        slope = 0
        intercept = 0
        # 列的数目过多则认为当前帧无效
        # if class_num <= 3:
            # x_axis = np.linspace(0, 1, 100)
            # colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
            # plt.ion()
            # plt.figure(1)
        obj_num = 0
        # 将点归类
        for center, label in zip(locs_list, clustering.labels_):
            # if label == -1:
            #     continue
            ret[label].append(center)
            obj_num += 1

        noise_rate = len(ret[-1]) / obj_num
        ret = ret[0:-1]

        # 定义待拟合直线方程，这里x归一化处理
        def f_1(x, k, b):
            return k * x + b

        # 拟合每一条检测出的直线
        for line in ret:
            # 此列点数小于3，则认为不可靠，不进行拟合
            # if len(line) <= 2:
            #     continue
            if noise_rate >= 0.4 or len(ret) <= 1:
                break
            xs = np.array([coord[0] for coord in line])
            ys = np.array([coord[1] for coord in line])
            # 最小二乘拟合
            popt, pcov = optimize.curve_fit(f_1, xs, ys)
            lines.append([popt[0] / camera_intrinsic["AspectRatio"], popt[1]])

            # slope = slope + popt[0]
            # intercept = intercept + popt[1]
            # print(popt)

            # plt.plot(x_axis, f_1(x_axis, popt[0], popt[1]))
        # 平均斜率
        # slope /= len(ret)
        # intercept /= len(ret)

        def takeDistance(elem):
            # 直线距离图像中心点在y轴上的投影
            return elem[0] * 0.5 + elem[1] - 0.5
        # 按照距离升序排序
        lines.sort(key = takeDistance)
        # if len(self.lines) >= 2:
            # print(takeDistance(self.lines[0]) * takeDistance(self.lines[1]))
        if len(lines) >= 2 and ( takeDistance(lines[0]) * takeDistance(lines[1]) <= 0 ):
            slope = (lines[0][0] + lines[1][0]) / 2
            intercept = (lines[0][1] + lines[1][1]) / 2
        else:
            # 保持不变
            slope = self.smoothed_slope
            intercept = self.smoothed_intercept
            # 归位
            # slope = 0
            # intercept = 0.5

        if self.slope_window.full():
            self.slope_window.get()
        self.slope_window.put(slope)
        if self.intercept_window.full():
            self.intercept_window.get()
        self.intercept_window.put(intercept)
        self.smoothed_slope = sum(self.slope_window.queue) / self.slope_window_size
        self.smoothed_intercept = sum(self.intercept_window.queue) / self.intercept_window_size

            # plt.xlim((0, 1))
            # plt.ylim((1, 0))
            # plt.show()
        # plt.cla()

        return self.smoothed_slope, self.smoothed_intercept, clustering.labels_, lines

    def global_guide_line_k_update(self, new_k):
        global guide_line_k
        guide_line_k = (1 / (1 - self.update_coefficient)) * guide_line_k + 1 / self.update_coefficient * new_k
        ret = guide_line_k
        return ret


def twoD_to_threeD(pos_in_img_space, cam_height):
    """
    2D转成3D
    :return: Position in the world coordinate system
    """

    # 此函数只是外部定义而已，大家可自行定义

    # (R T, 0 1)矩阵
    Trans = np.hstack((camera_intrinsic['R'], [[camera_intrinsic['T'][0]], [camera_intrinsic['T'][1]], [camera_intrinsic['T'][2]]]))
    tmp = [[0, 0, 0, 1]]
    Trans = np.concatenate((Trans, tmp), axis=0)
    # print(Trans)
    # 相机内参和相机外参 矩阵相乘
    temp = np.dot(camera_intrinsic['IntrinsicMatrix'], Trans)
    # 求伪逆
    Pp = np.linalg.pinv(temp)

    # 点（u, v, 1) 对应代码里的 [605,341,1]
    p1 = np.array(pos_in_img_space, np.float)

    # print("像素坐标系的点:", p1)

    X = np.dot(Pp, p1)

    # print("X:", X)

    # 与Zc相乘 得到世界坐标系的某一个点
    X1 = np.array(X[:3], np.float) * cam_height

    # print("X1:", X1)
    return X1


class InjuryRateCal:
    def __init__(self):
        self.injured = 0
        self.total = 0
        self.injury_rate = 0
        self.injury_rate_without_noise = 0
        self.injured_noise = 0

    def isInside(self, boxes, pos, label):
        bboxes_list = boxes[0].tolist()
        self.total += len(bboxes_list)
        for box, lab in zip(bboxes_list, label):
            if not (pos[0] <= box[0] or pos[1] >= box[2] or pos[1] <= box[1] or pos[1] >= box[3]):
                self.injured += 1
                if lab == -1:
                    self.injured_noise += 1

    def calculateInjuryRate(self):
        self.injury_rate = self.injured / self.total
        self.injury_rate_without_noise = (self.injured - self.injured_noise) / self.total
        print('Injured = \r\n', self.injured)
        print('Total = \r\n', self.total)
        print('Injured noise: \r\n', self.injured_noise)
        print('Injury rate = \r\n', self.injury_rate)
        print('Injury rate ignoring noises = \r\n', self.injury_rate_without_noise)
        return self.injury_rate


if __name__ == '__main__':
    twoD_to_threeD([250, 120, 1])