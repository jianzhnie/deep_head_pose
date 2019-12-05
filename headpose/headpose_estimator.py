import os
import torch
import numpy as np
import cv2
import time
from PIL import Image
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from math import cos, sin
import seaborn as sns
import matplotlib.pyplot as plt

import hopenet

class HeadPoseEstimator(object):
    """Provide headpose estimation APIs.
    """

    def __init__(self):

        self.test_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(240),
                        transforms.RandomCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])])

        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        self.model.load_state_dict(torch.load(os.path.join(
            os.path.dirname(__file__), "../models/resnet50_epoch_4.pkl")))

        self.model.cuda()
        self.model.eval()

    def estimate_headpose(self, image, bboxes):
        cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        head_images = []
        for i, bbox in enumerate(bboxes):
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])

            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = int(max(x_min, 0))
            y_min = int(max(y_min, 0))
            x_max = int(min(image.shape[1], x_max))
            y_max = int(min(image.shape[0], y_max))

            # Crop image
            img = cv2_image[y_min:y_max, x_min:x_max, :]

            # Transform
            img = self.test_transform(img)
            head_images.append(img)

        head_images = torch.stack(head_images, dim=0)
        head_images = head_images.cuda()

        yaws, pitchs, rolls = self.model(head_images)

        yaws_predicted = F.softmax(yaws, dim=1)
        pitchs_predicted = F.softmax(pitchs, dim=1)
        rolls_predicted = F.softmax(rolls, dim=1)

        # Get continuous predictions in degrees.
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda()

        yaws_predicted = torch.sum(yaws_predicted * idx_tensor, dim=1).cpu().detach().numpy() * 3 - 99
        pitchs_predicted = torch.sum(pitchs_predicted * idx_tensor, dim=1).cpu().detach().numpy() * 3 - 99
        rolls_predicted = torch.sum(rolls_predicted * idx_tensor, dim=1).cpu().detach().numpy() * 3 - 99

        headposes = np.array([[yaws_predicted[i], pitchs_predicted[i], rolls_predicted[i]] for i in range(len(bboxes))])
        return headposes

    @staticmethod
    def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
        # Input is a cv2 image
        # pose_params: (pitch, yaw, roll, tdx, tdy)
        # Where (tdx, tdy) is the translation of the face.
        # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

        p = pitch * np.pi / 180
        y = -(yaw * np.pi / 180)
        r = roll * np.pi / 180
        if tdx != None and tdy != None:
            face_x = tdx - 0.50 * size
            face_y = tdy - 0.50 * size
        else:
            height, width = img.shape[:2]
            face_x = width / 2 - 0.5 * size
            face_y = height / 2 - 0.5 * size

        x1 = size * (cos(y) * cos(r)) + face_x
        y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
        x2 = size * (-cos(y) * sin(r)) + face_x
        y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
        x3 = size * (sin(y)) + face_x
        y3 = size * (-cos(y) * sin(p)) + face_y

        # Draw base in red
        cv2.line(img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.line(img, (int(x2), int(y2)), (int(x2 + x1 - face_x), int(y2 + y1 - face_y)), (0, 0, 255), 3)
        cv2.line(img, (int(x1), int(y1)), (int(x1 + x2 - face_x), int(y1 + y2 - face_y)), (0, 0, 255), 3)
        # Draw pillars in blue
        cv2.line(img, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x1 + x3 - face_x), int(y1 + y3 - face_y)), (255, 0, 0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x2 + x3 - face_x), int(y2 + y3 - face_y)), (255, 0, 0), 2)
        cv2.line(img, (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
                 (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (255, 0, 0), 2)
        # Draw top in green
        cv2.line(img, (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
                 (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
        cv2.line(img, (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
                 (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x3 + x1 - face_x), int(y3 + y1 - face_y)), (0, 255, 0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x3 + x2 - face_x), int(y3 + y2 - face_y)), (0, 255, 0), 2)

        return img

    @staticmethod
    def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50):

        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

        return img