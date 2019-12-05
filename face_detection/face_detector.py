import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

# from .data.config import cfg
from .models.factory import build_net
from .utils.augmentations import to_chw_bgr


class FaceDetector(object):
    """Provide face detection APIs.
    """

    def __init__(self, cfg):

        self.draw_bboxes_color = cfg.DETECTION.DRAWING.BBOX_COLOR[::-1]
        self.draw_bboxes_width = cfg.DETECTION.DRAWING.BBOX_WIDTH

        self.net = build_net('test', model='vgg')
        self.net.load_state_dict(torch.load(os.path.join(
            os.path.dirname(__file__), "weights/dsfd_vgg_0.880.pth")))
        self.net.eval()

        self.device = cfg.DETECTION.CUDA_DEVICE
        self.net.to(self.device)

        self.threshold = cfg.DETECTION.THRESHOLD
        self.img_mean = np.array(cfg.img_mean)[
            :, np.newaxis, np.newaxis].astype('float32')
        self.input_size = cfg.DETECTION.INPUT_SIZE

        #torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def detect_faces(self, image):
        """Detect faces on the input image.

        Arguments:
            image {np.array} -- Image read by opencv.

        Returns:
            bboxes {np.array} -- Detected bboxes.
                Size: bbox_num x 5 (top_left_x, top_left_y, bottom_right_x, bottom_right_y, score)
        """

        # 缩小输入，不能检测出分辨率过大的人脸
        # print(self.input_size)
        small_image = cv2.resize(
            image, self.input_size, interpolation=cv2.INTER_LINEAR)

        x = to_chw_bgr(small_image)
        x = x.astype('float32')
        x -= self.img_mean
        x = x[[2, 1, 0], :, :]

        x = torch.from_numpy(x).unsqueeze(0)
        x = x.to(self.device)

        y = self.net(x)
        detections = y.data

        # 原图大小
        scale = torch.Tensor(
            [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

        result = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= self.threshold:
                score = detections[0, i, j, 0].cpu().numpy()

                # 把 [0,1] 的点映射回原图
                pt = (detections[0, i, j, 1:] *
                      scale).cpu().numpy()

                result.append(np.array([pt[0], pt[1], pt[2], pt[3], score]))

                j += 1

        return np.array(result)

    def draw_bboxes(self, img, bboxes, show_score=False, color=None):
        """Draw detected bboxes and their scores on the input image.

        Arguments:
            img {np.array} -- Image read by opencv.
            bboxes {np.array} -- Detected bboxes.
            color {tuple of 3 ints} -- Bbox color. If not specified, use default color in cofigs.
        Returns:
            result_img {np.array} -- Output image with bboxes.
        """
        result_img = img
        if len(bboxes) > 0:
            bboxes_int = np.int32(bboxes[:, :4])
            scores = np.round(bboxes[:, 4] * 100, 1)

            for index, pos in enumerate(list(bboxes_int)):
                # top_left, bottom_right
                cv2.rectangle(result_img, (pos[0], pos[1]), (pos[2], pos[3]),
                              self.draw_bboxes_color if color is None else color[::-1],
                              self.draw_bboxes_width)
                if show_score:
                    cv2.putText(result_img, str(scores[index]) + "%", (pos[0], pos[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.draw_bboxes_color,
                                self.draw_bboxes_width, cv2.LINE_AA)
        return result_img
